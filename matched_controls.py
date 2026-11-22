#!/usr/bin/env python3
"""Draw disjoint layer-matched channels and scale their signed Pub97 gaps to
the selected population's residual-write norm in the requested recipient.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from discovery import decoder_layers, load_model, sha256


def effective_columns(module, indices, recipient):
    """Columns of W_base, or W_base + sum(scale B A) for ordinary unmerged LoRA."""
    if getattr(module, "merged", False):
        raise ValueError("This control requires an unmerged base and adapter")
    if getattr(module, "fan_in_fan_out", False):
        raise ValueError("Expected a dense linear down projection in output × input layout")
    base = getattr(module, "base_layer", module)
    idx = torch.as_tensor(indices, dtype=torch.long, device=base.weight.device)
    columns = base.weight.detach().index_select(1, idx).float()
    if recipient == "medical" and hasattr(module, "lora_A"):
        if module.disable_adapters:
            raise ValueError("Medical control requested while adapters are disabled")
        for name in module.active_adapters:
            if name not in module.lora_A:
                continue
            if getattr(module, "use_dora", {}).get(name, False):
                raise ValueError("DoRA requires its own effective-weight calculation")
            a = module.lora_A[name].weight.detach().index_select(1, idx).float()
            b = module.lora_B[name].weight.detach().float()
            columns = columns + module.scaling[name] * (b @ a)
    return columns


@torch.no_grad()
def build_control(model, selection, layers, gaps, recipient, seed):
    if recipient not in ("base", "medical"):
        raise ValueError("Unknown recipient")
    if "control_recipient" in selection:
        raise ValueError("Expected a discovery selection as the input")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    layers = np.asarray(layers)
    gaps = np.asarray(gaps)
    if (layers.ndim != 1 or layers.dtype.kind not in "iu" or gaps.ndim != 2
            or gaps.shape[0] != layers.size or len(set(layers.tolist())) != len(layers)
            or not np.isfinite(gaps).all()):
        raise ValueError("Expected unique integer layers and finite layer × neuron mean gaps")
    if selection.get("layers", layers.tolist()) != layers.tolist():
        raise ValueError("Search layers differ between ranking and aggregate scores")
    modules = decoder_layers(model)
    layer_row = {int(layer): i for i, layer in enumerate(layers)}
    grouped = {}
    for row in selection["selected"]:
        layer, channel = row["layer"], row["neuron"]
        if (type(layer) is not int or layer not in layer_row or not 0 <= layer < len(modules)
                or type(channel) is not int or not 0 <= channel < gaps.shape[1]):
            raise ValueError("Selected coordinate is outside the searched model")
        if float(row["mean_activation_gap"]) != float(gaps[layer_row[layer], channel]):
            raise ValueError("Selected mean gap does not match its aggregate score file")
        grouped.setdefault(layer, []).append(channel)
    if not grouped:
        raise ValueError("Empty selection")

    rng = np.random.default_rng(seed)
    controls, audit = [], []
    for layer, chosen in sorted(grouped.items()):
        down = modules[layer].mlp.down_proj
        width = down.in_features
        if width != gaps.shape[1] or len(set(chosen)) != len(chosen):
            raise ValueError("Model width mismatch or duplicate chosen channel")
        pool = np.setdiff1d(np.arange(width), chosen, assume_unique=True)
        if pool.size < len(chosen):
            raise ValueError("Insufficient disjoint channels to match layer count")
        sampled = rng.choice(pool, size=len(chosen), replace=False)
        all_idx = chosen + sampled.tolist()
        columns = effective_columns(down, all_idx, recipient)
        gap = torch.tensor(gaps[layer_row[layer], all_idx], device=columns.device, dtype=torch.float32)
        k = len(chosen)
        target = torch.linalg.vector_norm(columns[:, :k] @ gap[:k])
        raw_norm = torch.linalg.vector_norm(columns[:, k:] @ gap[k:])
        if not torch.isfinite(target) or not torch.isfinite(raw_norm) or raw_norm <= 1e-8:
            raise ValueError(f"Undefined write-norm ratio at layer {layer}")
        scale = target / raw_norm
        scaled = gap[k:] * scale
        after = torch.linalg.vector_norm(columns[:, k:] @ scaled)
        torch.testing.assert_close(after, target, rtol=3e-5, atol=1e-6)
        audit.append(dict(layer=layer, count=k, selected_norm=target.item(),
                          control_norm_before=raw_norm.item(), control_norm_after=after.item(),
                          scale=scale.item()))
        for channel, value in zip(sampled.tolist(), scaled.tolist()):
            controls.append(dict(layer=layer, neuron=channel, mean_activation_gap=value,
                                 unscaled_mean_gap=float(gaps[layer_row[layer], channel])))
    result = {k: selection[k] for k in ("base", "adapter", "base_revision", "adapter_revision", "panel", "part",
                                        "split_seed", "eval_size", "discovery_question_ids",
                                        "input_sha256", "source_sha256")
              if k in selection}
    result.update(schema_version=1, selected=controls, control_recipient=recipient,
                  control_seed=seed, control_protocol="uniform_disjoint_layer_count_and_write_norm",
                  norm_audit=audit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=Path, required=True, help="Discovery selected_neurons.json")
    parser.add_argument("--scores", type=Path, required=True, help="Matching discovery all_scores.npz")
    parser.add_argument("--recipient", required=True, choices=["base", "medical"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", type=Path, required=True, help="New control JSON for steering --neurons")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    selection = json.loads(args.neurons.read_text())
    if selection.get("scores_sha256") != sha256(args.scores):
        parser.error("The score file does not belong to this discovery selection")
    torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = load_model(selection["base"], selection["adapter"], args.device,
                          dtype=torch.float32, base_revision=selection.get("base_revision"),
                          adapter_revision=selection.get("adapter_revision"),
                          local_files_only=args.local_files_only)
    with np.load(args.scores, allow_pickle=False) as arrays:
        result = build_control(model, selection, arrays["layers"], arrays["mean_gap"],
                               args.recipient, args.seed)
    result.update(selection_sha256=sha256(args.neurons), scores_sha256=sha256(args.scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(result['selected'])} control channels for recipient {args.recipient}")


if __name__ == "__main__":
    main()
