"""Output-distribution and shared-template penalties for adapter training on Pub97."""
import torch

from discovery import decoder_layers


def _require_adapter(model):
    if not getattr(model, "active_adapters", None) or not getattr(model, "has_active_enabled_adapter", False):
        raise ValueError("This training comparison requires an active, enabled adapter")


def kl_to_base(model, batch, logits):
    """Mean D_KL(p_adapted || p_base) in nats per attended position, including the prompt.

    logits[b,t] is the next-token distribution after input position t. The
    mask is at that same position, independently of the supervised-answer mask.
    """
    _require_adapter(model)
    mask = batch["attention_mask"].bool()
    if logits.ndim != 3 or logits.shape[:2] != mask.shape or not mask.any():
        raise ValueError("KL requires aligned logits and a nonempty token mask")
    was_training = model.training
    try:
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            base = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            base_logp = base[mask].float().log_softmax(-1)
    finally:
        model.train(was_training)
    logp = logits[mask].float().log_softmax(-1)
    return (logp.exp() * (logp - base_logp)).sum(-1).mean()


class SharedPrefixKV:
    """TReFT's relative K/V projection drift on fixed tokens before user content.

    training_prefixes contains each unpadded native user chat sequence (full
    supervised sequences are also accepted). The model must have a frozen
    base. Call loss() once per optimizer update for the shared prefix.
    """

    def __init__(self, model, tokenizer, training_prefixes):
        _require_adapter(model)
        core = model.get_base_model()
        if core.config.model_type not in {"qwen2", "llama"}:
            raise ValueError("Shared-prefix KV readout supports Qwen2.5 and Llama")
        marker = "<<|USER_SPLIT_MARKER|>>"
        text = tokenizer.apply_chat_template([{"role": "user", "content": marker}],
                                             tokenize=False, add_generation_prompt=False)
        if text.count(marker) != 1:
            raise ValueError("Cannot locate the user-content boundary in the native chat template")
        tokens = tokenizer.encode(text.partition(marker)[0], add_special_tokens=False)
        if not tokens:
            raise ValueError("The native chat template has no shared prefix")
        count = 0
        for ids in training_prefixes:
            if list(ids[:len(tokens)]) != tokens:
                raise ValueError("Template prefix token IDs differ across supervised examples")
            count += 1
        if not count:
            raise ValueError("Supply supervised token sequences to verify the shared prefix")
        self.model = model
        self.prefix_ids = list(tokens)
        self.tokens = torch.tensor([tokens], dtype=torch.long, device=model.get_input_embeddings().weight.device)
        self.backbone = core.model
        self.layers = decoder_layers(model)
        was_training = model.training
        try:
            model.eval()
            with model.disable_adapter(), torch.no_grad():
                self.reference = list(self._projections())
        finally:
            model.train(was_training)

    def _projections(self):
        hidden = self.backbone(input_ids=self.tokens, use_cache=False, output_hidden_states=True,
                               return_dict=True).hidden_states
        # The released TReFT implementation applies each affine projection to
        # the layer-input residual. Use the complete concatenated KV width for
        # each token, before head reshaping or rotary positional embeddings.
        for layer, state in zip(self.layers, hidden[:-1], strict=True):
            attn = layer.self_attn
            yield attn.k_proj(state).float(), attn.v_proj(state).float()

    def loss(self):
        _require_adapter(self.model)
        was_training = self.model.training
        try:
            self.model.eval()
            layers = []
            for reference, current in zip(self.reference, self._projections(), strict=True):
                terms = []
                for base, active in zip(reference, current, strict=True):
                    magnitude = torch.linalg.vector_norm(base, dim=-1).clamp_min(1e-12)
                    shift = torch.linalg.vector_norm(active - base, dim=-1)
                    terms.append((shift / magnitude).square().mean())
                layers.append(terms[0] + terms[1])
            return torch.stack(layers).mean()
        finally:
            self.model.train(was_training)
