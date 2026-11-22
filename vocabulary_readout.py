#!/usr/bin/env python3
"""Read signed Pub97 neuron writes through a pre-fitted lens and the model's vocabulary head."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from discovery import decoder_layers, sha256
from methods.readouts import decode_write, transport_writes
from methods.vocabulary_controls import centered_fastica, dose_weighted_writes, matched_random_writes
from methods.vocabulary_inputs import load_readout_model, load_transport, readout_modules, write_columns
from methods.vocabulary_ranks import control_rank_rows, ranked_tokens
from pub97 import SOURCE_SHA256, selected_layers


def layer_and_joint(vectors, method):
    """Combine layer directions before the vocabulary projection."""
    if len(vectors) > 1:
        yield "joint_" + method, torch.stack(list(vectors.values())).sum(0)
    for layer, direction in vectors.items():
        yield f"layer_{layer}_{method}", direction


@torch.inference_mode()
def run(args, output):
    selection = json.loads(args.neurons.read_text(encoding="utf-8"))
    groups = selected_layers(selection, args.layers)
    checkpoint = {k: selection[k] for k in ("base", "adapter", "base_revision", "adapter_revision")}
    if args.scores:
        if sha256(args.scores) != selection["scores_sha256"]:
            raise ValueError("The score archive does not belong to this discovery population")
        with np.load(args.scores, allow_pickle=False) as archive:
            search_layers, gaps = archive["layers"], archive["mean_gap"]
        if selection["layers"] != search_layers.tolist():
            raise ValueError("The score archive uses different discovery layers")

    model, tokenizer = load_readout_model(**checkpoint, device=args.device, write_weights=args.write_weights,
                                          local_files_only=args.local_files_only)
    norm, head = readout_modules(model)
    matrices = None
    lens_info = None
    if args.lens:
        matrices, lens_info = load_transport(args.lens, groups, norm.weight.numel(),
                                             model.get_base_model().config.model_type, args.method,
                                             base=checkpoint["base"], base_revision=checkpoint["base_revision"])
        matrices = {layer: mat.to(norm.weight.device, torch.float32) for layer, mat in matrices.items()}

    def columns(layer, ids):
        return write_columns(model, layer, ids, args.write_weights)

    def transport(layer, writes):
        return writes if matrices is None else transport_writes(writes, matrices[layer])

    raw_vectors, carried_vectors, ica_rows = {}, {}, []
    directions = {}
    for layer, rows in groups.items():
        ids = [r["neuron"] for r in rows]
        raw = dose_weighted_writes(columns(layer, ids), [r["mean_activation_gap"] for r in rows], args.alpha)
        carried = transport(layer, raw)
        raw_vectors[layer] = raw.sum(0)
        carried_vectors[layer] = carried.sum(0)
        if args.per_neuron:
            for neuron, vector in zip(ids, carried, strict=True):
                directions[f"neuron_{layer}_{neuron}_{args.method}"] = vector
        if args.ica_components:
            ica_rows.append(carried.cpu())

    directions.update({"selected_" + name: vec for name, vec in layer_and_joint(raw_vectors, "logit")})
    if matrices is not None:
        directions.update({"selected_" + name: vec for name, vec in layer_and_joint(carried_vectors, args.method)})
    controls = []
    if args.scores:
        modules = decoder_layers(model)
        if gaps.ndim != 2 or any(modules[layer].mlp.down_proj.in_features != gaps.shape[1] for layer in groups):
            raise ValueError("The discovery gap table must cover the full channel width of each requested layer")
        random_by_seed = defaultdict(dict)
        selected = [r for rows in groups.values() for r in rows]
        for audit, writes in matched_random_writes(selected, search_layers, gaps, columns, alpha=args.alpha):
            controls.append(audit)
            layer = audit["layer"]
            random_by_seed[audit["seed"]][layer] = transport(layer, writes).sum(0)
        for seed, vectors in random_by_seed.items():
            for name, vec in layer_and_joint(vectors, args.method):
                directions[f"random_{seed}_{name}"] = vec

    fits = []
    if ica_rows:
        observations = torch.cat(ica_rows)
        for count in args.ica_components:
            for seed in args.ica_seeds:
                components, fit = centered_fastica(observations, count, seed=seed)
                fits.append(fit)
                for index, direction in enumerate(components):
                    directions[f"ica_{count}_seed_{seed}_component_{index}_{args.method}"] = direction

    scores, selected_tables = {}, {}
    with (output / "tokens.csv").open("w", encoding="utf-8", newline="") as table:
        fields = ["group", "view", "end", "position", "raw_rank", "token_id", "token", "score"]
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for group, direction in directions.items():
            z = decode_write(direction, norm, head).cpu().numpy()
            scores[group] = z
            rows = list(ranked_tokens(z, tokenizer, args.top_k, args.filtered_top_k))
            writer.writerows(dict(group=group, **row) for row in rows)
            if controls and group.startswith("selected_") and group.endswith("_" + args.method):
                selected_tables[group.removeprefix("selected_")] = rows
    np.savez_compressed(output / "scores.npz", **scores)
    artifacts = ["scores.npz", "tokens.csv"]
    if controls:
        artifacts.append("control_ranks.csv")
        with (output / "control_ranks.csv").open("w", encoding="utf-8", newline="") as table:
            fields = ["group", "end", "token_id", "token", "selected_rank", "best_random_rank",
                      "random_top50"] + [f"seed_{s}_rank" for s in range(10)]
            writer = csv.DictWriter(table, fieldnames=fields)
            writer.writeheader()
            for group, rows in selected_tables.items():
                random_scores = {s: scores[f"random_{s}_{group}"] for s in range(10)}
                writer.writerows(dict(group="selected_" + group, **row)
                                 for row in control_rank_rows(rows, random_scores))
    manifest = dict(
        panel="pub97", source_sha256=SOURCE_SHA256, selection_sha256=sha256(args.neurons),
        split_seed=selection["split_seed"], eval_size=selection["eval_size"],
        discovery_question_ids=selection["discovery_question_ids"], **checkpoint,
        method=args.method, transport=lens_info, write_weights=args.write_weights, alpha=args.alpha,
        per_neuron=args.per_neuron,
        coordinates=[r for rows in groups.values() for r in rows],
        norm_dtype=str(norm.weight.dtype), head_dtype=str(head.weight.dtype),
        vocab_rows=head.weight.shape[0], top_k=args.top_k, filtered_top_k=args.filtered_top_k,
        discovery_scores_sha256=selection["scores_sha256"] if args.scores else None,
        random_controls=controls, ica_fits=fits,
        output_sha256={name: sha256(output / name) for name in artifacts})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=Path, required=True, help="Pub97 discovery selected_neurons.json")
    parser.add_argument("--method", choices=("jacobian", "jpp", "logit"), required=True)
    parser.add_argument("--lens", type=Path, help="Local .pt for the chosen pre-fitted transport")
    parser.add_argument("--layers", type=int, nargs="+", help="Include these occupied layers (default: all)")
    parser.add_argument("--per-neuron", action="store_true", help="Also read each selected channel individually")
    parser.add_argument("--scores", type=Path, help="Discovery all_scores.npz, to add ten matched random controls")
    parser.add_argument("--write-weights", choices=("base", "effective"), default="base")
    parser.add_argument("--alpha", type=float, default=2., help="Dose times the signed mean activation gap")
    parser.add_argument("--ica-components", type=int, nargs="+", help="Optional component counts for centred FastICA")
    parser.add_argument("--ica-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top-k", type=int, default=50, help="Unfiltered table slots per end")
    parser.add_argument("--filtered-top-k", type=int, default=30, help="Unicode-filtered slots per end; 0 disables")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New directory for vocabulary readout")
    args = parser.parse_args(argv)
    if (args.method != "logit") != bool(args.lens):
        parser.error("Supply --lens for jacobian or jpp; logit uses the identity transform")
    if (not math.isfinite(args.alpha) or args.alpha == 0 or args.top_k < 1 or args.filtered_top_k < 0
            or args.output.exists()):
        parser.error("Specify a finite nonzero dose, valid table lengths and a new output directory")
    if args.ica_components and (not args.lens or min(args.ica_components) < 1
                                or len(set(args.ica_components)) != len(args.ica_components)
                                or len(set(args.ica_seeds)) != len(args.ica_seeds)
                                or any(not 0 <= s < 2**32 for s in args.ica_seeds)):
        parser.error("FastICA requires transported writes, distinct positive component counts and distinct uint32 seeds")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as staging:
        staged = Path(staging) / "readout"
        staged.mkdir()
        run(args, staged)
        staged.rename(args.output)
    print(f"Saved vocabulary readout to {args.output}")


if __name__ == "__main__":
    main()
