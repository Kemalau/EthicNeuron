#!/usr/bin/env python3
"""Apply a sparse mean-gap shift to nonpadding prompt and cached decoding activations."""
from __future__ import annotations

import math

import torch


class SparseSteering:
    """One generation session: all real prompt tokens + N cached forward calls.

    A new context is required for every model.generate() call. Beam search,
    speculative decoding and uncached multi-token decoding are unsupported.
    """
    def __init__(self, layers, selected, alpha, prompt_mask, decode_steps=32):
        if not math.isfinite(alpha) or type(decode_steps) is not int or decode_steps < 0:
            raise ValueError('alpha must be finite; decode_steps must be nonnegative')
        if prompt_mask.ndim != 2 or not prompt_mask.bool().any(dim=1).all():
            raise ValueError('Expected a nonempty [batch, prompt] attention mask')
        if not selected:
            raise ValueError('The selection is empty')
        self.layers, self.alpha = layers, alpha
        self.mask, self.decode_steps = prompt_mask.bool(), decode_steps
        self.parts, self.handles, self.calls = {}, [], {}
        self.edited_calls = {}
        seen = set()
        for row in selected:
            layer, neuron, gap = row['layer'], row['neuron'], row['mean_activation_gap']
            if type(layer) is not int or type(neuron) is not int:
                raise ValueError('Layer and neuron indices must be integers')
            if not 0 <= layer < len(layers) or not math.isfinite(gap):
                raise ValueError('Invalid layer or activation gap')
            width = layers[layer].mlp.down_proj.in_features
            if not 0 <= neuron < width or (layer, neuron) in seen:
                raise ValueError('Duplicate or out-of-range neuron')
            seen.add((layer, neuron))
            self.parts.setdefault(layer, []).append((neuron, gap))

    def __enter__(self):
        if self.handles or self.calls:
            raise RuntimeError('Create a fresh SparseSteering context per generation')
        try:
            for layer, rows in self.parts.items():
                self.calls[layer] = self.edited_calls[layer] = 0
                def shift(module, args, layer=layer, rows=rows):
                    self.calls[layer] += 1
                    call, x = self.calls[layer], args[0]
                    if x.ndim != 3 or x.shape[0] != self.mask.shape[0]:
                        raise ValueError('Unexpected batch; use num_beams=1 and one return sequence')
                    if call == 1:
                        if x.shape[:2] != self.mask.shape:
                            raise ValueError('Prefill shape differs from prompt mask')
                        valid = self.mask.to(x.device)
                    else:
                        if x.shape[1] != 1:
                            raise ValueError('Steering requires cached single-token decoding')
                        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                    if call > 1 + self.decode_steps or self.alpha == 0:
                        return None
                    index = torch.tensor([r[0] for r in rows], device=x.device)
                    delta = torch.tensor([r[1] for r in rows], dtype=torch.float32,
                                         device=x.device) * self.alpha
                    out = x.clone()
                    before = x[..., index]
                    out[..., index] = torch.where(valid[..., None],
                                                   (before.float() + delta).to(x.dtype), before)
                    self.edited_calls[layer] += 1
                    return (out, *args[1:])
                self.handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(shift))
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, *exc):
        self.close()
