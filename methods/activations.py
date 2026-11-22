"""Selected-channel activations and standardized activation anchoring on Pub97 prompts."""

from collections import defaultdict
from contextlib import ExitStack
import math

import torch


class SelectedChannels:
    """Capture the input to each selected MLP down projection in selection-file order."""

    def __init__(self, down_projections, selected):
        self.layers = list(down_projections)
        self.channels = defaultdict(list)
        seen = set()
        for position, row in enumerate(selected):
            layer, neuron = row["layer"], row["neuron"]
            if type(layer) is not int or not 0 <= layer < len(self.layers):
                raise ValueError("Layer outside model")
            if type(neuron) is not int or not 0 <= neuron < self.layers[layer].in_features:
                raise ValueError("Channel outside down projection")
            if (layer, neuron) in seen:
                raise ValueError("Duplicate channel")
            self.channels[layer].append((position, neuron))
            seen.add((layer, neuron))
        if not seen:
            raise ValueError("Empty selection")
        # Capture each layer once, then restore the caller's global coordinate order.
        positions = [p for layer in self.channels for p, _ in self.channels[layer]]
        self.order = sorted(range(len(positions)), key=positions.__getitem__)

    def capture(self, model, batch, *, boundary=False):
        """Return [nonpad tokens, channels] or [prompts, channels] at the final nonpad token."""
        ids, mask = batch["input_ids"], batch["attention_mask"].bool()
        if ids.ndim != 2 or ids.shape != mask.shape or not ids.numel() or not mask.any(1).all():
            raise ValueError("Expected a nonempty token batch with at least one real token in each prompt")
        device = model.get_input_embeddings().weight.device
        ids, mask = ids.to(device), mask.to(device)
        positions = mask
        if boundary:
            last = torch.arange(mask.shape[1], device=device).expand_as(mask).masked_fill(~mask, -1).amax(1)
            positions = torch.zeros_like(mask)
            positions.scatter_(1, last[:, None], True)

        values = {}

        def capture_layer(layer):
            indices = [i for _, i in self.channels[layer]]

            def hook(module, args):
                x = args[0]
                if x.shape[:2] != mask.shape or layer in values:
                    raise ValueError("Expected one full prompt forward per layer")
                values[layer] = x[..., indices][positions.to(x.device)].to(device)
            return hook

        with ExitStack() as handles:
            for layer in self.channels:
                handle = self.layers[layer].register_forward_pre_hook(capture_layer(layer))
                handles.callback(handle.remove)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        return torch.cat([values[l] for l in self.channels], dim=-1)[:, self.order]


class ActivationAnchor:
    """Penalize movement away from the frozen adapter-disabled reference at valid prompt tokens."""

    def __init__(self, model, channels, prompt_batches, *, sigma_floor=1e-3):
        if not math.isfinite(sigma_floor) or sigma_floor <= 0:
            raise ValueError("Expected a positive finite standard deviation floor")
        self.model = model
        self.channels = channels
        total = square = None
        count = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), model.disable_adapter():
                for batch in prompt_batches:
                    values = channels.capture(model, batch).double()
                    s, ss = values.sum(0), values.square().sum(0)
                    total = s if total is None else total + s
                    square = ss if square is None else square + ss
                    count += values.shape[0]
        finally:
            model.train(was_training)
        if count < 2:
            raise ValueError("At least two real calibration tokens are required")
        variance = (square / count - (total / count).square()).clamp_min(0)
        self.sigma = variance.sqrt().clamp_min(sigma_floor).float()
        if not self.sigma.isfinite().all():
            raise ValueError("Invalid calibration activations")

    def loss(self, batch):
        """Return a differentiable scalar for the specified target-free anchor minibatch."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self.model.disable_adapter():
                reference = self.channels.capture(self.model, batch)
            current = self.channels.capture(self.model, batch)
            return ((current.float() - reference.float()) / self.sigma).square().mean()
        finally:
            self.model.train(was_training)
