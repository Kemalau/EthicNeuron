"""Measure fixed Pub97 neuron populations across a supplied series of adapter checkpoints."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import torch

from discovery import decoder_layers, load_model, sha256
from generate import batch_seed, prefix, run_batch
from methods.activations import SelectedChannels
from methods.readouts import checkpoint_projection
from pub97 import MANIFEST_SHA256, SOURCE_SHA256, load_pub97, select_split, selected_layers


TRAINING_FIELDS = ("arm", "beta", "kl_coefficient", "treft_coefficient", "responses_sha256", "anchor_members", "anchor_question_ids",
                   "control_seed", "batch_size", "accumulation", "epochs", "optimizer", "learning_rate",
                   "weight_decay", "gradient_clip_norm", "warmup_steps", "schedule", "sequence_cap",
                   "rslora_rank", "lora_alpha", "lora_dropout", "sft_reduction")


def read_checkpoints(path, population, selection_hash):
    """Bind each nonzero observation to one training seed, actual step and adapter."""
    manifest = json.loads(path.read_text())
    if any(manifest[k] != population[k] for k in ("base", "base_revision")):
        raise ValueError("Checkpoint manifest base differs from the discovery model")
    expected = {key: population[key] for key in ("base", "base_revision", "panel", "source_sha256",
                                                 "split_seed", "eval_size", "discovery_question_ids")}
    expected["selection_sha256"] = selection_hash
    expected["manifest_sha256"] = MANIFEST_SHA256
    records = {}
    training = {}
    for row in manifest["checkpoints"]:
        seed, step = row["seed"], row["step"]
        if type(seed) is not int or seed < 0 or type(step) is not int or step <= 0:
            raise ValueError("Specify a nonnegative training seed and a positive saved optimizer step")
        if (seed, step) in records:
            raise ValueError("Duplicate training seed and step")
        if "seed" in manifest and seed != manifest["seed"]:
            raise ValueError("Checkpoint seed differs from its training manifest")
        for key, value in expected.items():
            if any(key in source and source[key] != value for source in (manifest, row)):
                raise ValueError(f"Checkpoint training provenance differs at {key}")
        for key in (*TRAINING_FIELDS, "calibration_question_ids"):
            if key in row and key in manifest and row[key] != manifest[key]:
                raise ValueError(f"Mixed checkpoint training settings at {key}")
        source = {key: row.get(key, manifest.get(key)) for key in (*TRAINING_FIELDS, "calibration_question_ids")}
        if records and training != source:
            raise ValueError("Use separate manifests for different training arms and calibration pools")
        training = source
        adapter = row["adapter"]
        revision = row.get("adapter_revision")
        if not isinstance(adapter, str) or not adapter.strip():
            raise ValueError("Checkpoint requires an adapter path or Hub ID")
        local = path.parent / Path(adapter).expanduser()
        identity = dict(adapter=adapter, adapter_revision=revision)
        if local.is_dir():
            identity["adapter"] = str(local.resolve())
            identity["adapter_sha256"] = sha256(local / "adapter_model.safetensors")
            identity["config_sha256"] = sha256(local / "adapter_config.json")
        elif revision is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Missing local checkpoint or immutable Hub revision for {adapter}")
        records[seed, step] = identity
    if not records:
        raise ValueError("No saved checkpoints in manifest")
    manifest.update({k: v for k, v in training.items() if v is not None})
    return manifest, sorted(records.items())


def population_with_control(population, archive, seed):
    groups = selected_layers(population, None)
    layers, gaps = archive["layers"], archive["mean_gap"]
    if (layers.ndim != 1 or layers.dtype.kind not in "iu" or len(set(layers)) != len(layers)
            or layers.tolist() != population["layers"] or gaps.ndim != 2
            or gaps.shape[0] != len(layers) or not np.isfinite(gaps).all()):
        raise ValueError("Invalid discovery layer and signed mean gap arrays")
    rng = np.random.default_rng(seed)
    selected, control = [], []
    for layer, rows in groups.items():
        if layer not in layers:
            raise ValueError("Selected layer is outside the discovery search")
        table = gaps[layers.tolist().index(layer)]
        ids = [row["neuron"] for row in rows]
        if max(ids) >= len(table) or any(table[i] != row["mean_activation_gap"] for i, row in zip(ids, rows)):
            raise ValueError("Population gap differs from the discovery archive")
        available = np.setdiff1d(np.arange(len(table)), ids)
        if len(available) < len(ids):
            raise ValueError("Too few disjoint channels for the layer-matched control")
        selected.extend(rows)
        for i in sorted(rng.choice(available, size=len(ids), replace=False).tolist()):
            control.append(dict(layer=layer, neuron=i, mean_activation_gap=float(table[i])))
    return selected, control


def capture_setup(model, coordinates):
    core = model.get_base_model()
    backbone = core.model
    if core.config.model_type in {"qwen3_5", "gemma3"}:
        backbone = backbone.language_model
    channels = SelectedChannels([layer.mlp.down_proj for layer in decoder_layers(model)], coordinates)
    return backbone, channels


def capture_one(backbone, channels, tokens, *, boundary):
    ids = torch.tensor([tokens], dtype=torch.long)
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    return channels.capture(backbone, batch, boundary=boundary).double().cpu()


def training_sigma(backbone, channels, prefixes, calibration_ids):
    """Population SD over real template/prompt tokens. A repeated question counts per row."""
    total = square = None
    n_tokens = 0
    for qid, multiplicity in Counter(calibration_ids).items():
        values = capture_one(backbone, channels, prefixes[qid], boundary=False)
        s = values.sum(0) * multiplicity
        ss = values.square().sum(0) * multiplicity
        total = s if total is None else total + s
        square = ss if square is None else square + ss
        n_tokens += len(values) * multiplicity
    if n_tokens < 2:
        raise ValueError("Need at least two calibration tokens")
    sigma = (square / n_tokens - (total / n_tokens).square()).clamp_min(0).sqrt().clamp_min(1e-3)
    if not sigma.isfinite().all():
        raise ValueError("Nonfinite calibration values")
    return sigma, n_tokens


def generate_checkpoint(stream, model, tok, questions, seed, step, identity, binding, args, split):
    """Use common question/draw streams across steps within each training seed."""
    condition = f"trajectory_seed{seed}_step{step}"
    recipient = "base" if step == 0 else "medical"
    tasks = [dict(q, pair_id=f"pub97:evaluation:{q['question_id']}:{i}", sample_idx=i)
             for q in questions for i in range(args.behavior_draws)]
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start:start + args.batch_size]
        key = batch_seed(seed, batch[0]["pair_id"])
        for row in run_batch(model, tok, [], batch, recipient, 0., key, args.max_new_tokens):
            row.update(identity, panel="pub97", part="evaluation", split_seed=split["seed"],
                       eval_size=split["eval_size"], condition=condition, training_seed=seed, step=step,
                       id=f"{condition}:{row['pair_id']}", binding=dict(binding, **identity, master_seed=seed))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@torch.inference_mode()
def measure(args, output):
    population = json.loads(args.neurons.read_text())
    if sha256(args.scores) != population["scores_sha256"]:
        raise ValueError("The score archive is not bound to this discovery population")
    with np.load(args.scores, allow_pickle=False) as archive:
        selected, control = population_with_control(population, archive, args.control_seed)
    selection_hash = sha256(args.neurons)
    manifest, records = read_checkpoints(args.checkpoints, population, selection_hash)
    panel = load_pub97(args.source)
    split = dict(seed=population["split_seed"], eval_size=population["eval_size"])
    training = select_split(panel, "discovery", **split)
    monitor = select_split(panel, "evaluation", **split)
    if not monitor:
        raise ValueError("Reserve a held-out Pub97 partition before discovery")
    training_ids = [q["question_id"] for q in training]
    monitor_ids = [q["question_id"] for q in monitor]
    cal_ids = manifest.get("calibration_question_ids", training_ids)
    if not isinstance(cal_ids, list) or not cal_ids or not set(cal_ids) <= set(training_ids):
        raise ValueError("Calibration requires a nonempty list of Pub97 discovery-partition question IDs")
    if not set(manifest.get("anchor_question_ids", [])) <= set(training_ids):
        raise ValueError("Anchor training questions overlap the held-out monitor or fall outside Pub97")
    calibration_source = ("declared_training_prompts" if "calibration_question_ids" in manifest
                          else "discovery_partition_prompts")
    coordinates = selected + control
    gap = torch.tensor([r["mean_activation_gap"] for r in coordinates], dtype=torch.float64)
    k = len(selected)
    base = {k: population[k] for k in ("base", "base_revision")}
    binding = dict(manifest_sha256=MANIFEST_SHA256, source_sha256=SOURCE_SHA256,
                   selection_sha256=selection_hash, panel_sha256=hashlib.sha256(
                       "\n".join(monitor_ids).encode()).hexdigest(), samples=args.behavior_draws,
                   temperature=1., top_p=1., top_k=0, repetition_penalty=1.,
                   max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, dtype="bfloat16")
    results = []
    out = (output / "generations.jsonl").open("w", encoding="utf-8") if args.behavior_draws else nullcontext()
    with out as stream:
        for index, ((seed, step), checkpoint) in enumerate(records):
            model, tok = load_model(**base, adapter=checkpoint["adapter"], adapter_revision=checkpoint["adapter_revision"],
                                    device=args.device, dtype=torch.bfloat16, local_files_only=args.local_files_only)
            backbone, channels = capture_setup(model, coordinates)
            if index == 0:
                def render(q):
                    return prefix(tok, q, model.get_base_model().config.model_type)
                prefixes = {q["question_id"]: render(q) for q in training + monitor}
                with model.disable_adapter():
                    sigma, n_tokens = training_sigma(backbone, channels, prefixes, cal_ids)
                    reference = torch.cat([capture_one(backbone, channels, prefixes[qid], boundary=True)
                                           for qid in monitor_ids])
            if index == 0 or seed != records[index-1][0][0]:
                results.append(dict(seed=seed, step=0, selected=0., random=0., condition=f"trajectory_seed{seed}_step0"))
                if stream:
                    # Decoder is adapter-disabled for the original checkpoint.
                    identity = dict(base, adapter=None, adapter_revision=None)
                    generate_checkpoint(stream, model, tok, monitor, seed, 0, identity, binding, args, split)
            current = torch.cat([capture_one(backbone, channels, prefixes[qid], boundary=True) for qid in monitor_ids])
            scores = {}
            for name, ix in (("selected", slice(None, k)), ("random", slice(k, None))):
                scores[name] = checkpoint_projection(current[:, ix], reference[:, ix], gap[ix], sigma[ix]).item()
            results.append(dict(seed=seed, step=step, condition=f"trajectory_seed{seed}_step{step}", **checkpoint, **scores))
            if stream:
                generate_checkpoint(stream, model, tok, monitor, seed, step, dict(base, **checkpoint), binding, args, split)
            del model, backbone, channels, tok
    report = dict(panel="pub97", **base, split=split, selection_sha256=binding["selection_sha256"],
                  source_sha256=SOURCE_SHA256, manifest_sha256=MANIFEST_SHA256,
                  checkpoint_manifest_sha256=sha256(args.checkpoints), scores_sha256=population["scores_sha256"],
                  training_settings={k: manifest[k] for k in TRAINING_FIELDS if k in manifest},
                  calibration_source=calibration_source, calibration_question_ids=cal_ids, calibration_token_count=n_tokens,
                  sigma_floor=1e-3, monitor_question_ids=monitor_ids, control_seed=args.control_seed,
                  selected=selected, random=control, sigma=sigma.tolist(), checkpoints=results)
    (output / "trajectory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "neurons", "scores", "checkpoints", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--control-seed", type=int, default=42)
    p.add_argument("--behavior-draws", type=int, default=0, help="Optional unsteered completions/question/checkpoint")
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=2, help="Behavior generation batch size")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)
    if args.output.exists() or min(args.control_seed, args.behavior_draws) < 0 or min(args.batch_size, args.max_new_tokens) < 1:
        p.error("Choose a fresh output directory, a nonnegative seed/draw count and positive batch size/token cap")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        target = Path(directory) / "trajectory"
        target.mkdir()
        measure(args, target)
        target.rename(args.output)


if __name__ == "__main__":
    main()
