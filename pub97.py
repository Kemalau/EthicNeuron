"""Resolve Pub97 by exact source identity and partition its questions reproducibly."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
MANIFEST_SHA256 = "89994c06aefe6d90cbc1c4661f069974053500aa8bbfb8a5a362dbd1dfd10b7c"
SOURCE_SHA256 = "73a91f706e4f7eb314628f22b419e62baff1311f1263443abd0dd880344d7e30"
DEFAULT_SEED = 42
DEFAULT_EVAL_SIZE = 20


def _csv_rows(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {Path(path).name}. See data/README.md for the pinned input.")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


def panel_manifest():
    """Read the verified Pub97 identifiers and original UTF-8 prompt digests in source order."""
    entries = _csv_rows(DATA / "pub97.csv", MANIFEST_SHA256)
    for entry in entries:
        entry["source_row"] = int(entry["source_row"])
    return entries


def load_pub97(source):
    """Load the exact 97 prompts from a local, pinned upstream CSV. Preserve whitespace."""
    upstream = _csv_rows(source, SOURCE_SHA256)
    result = []
    for entry in panel_manifest():
        source_row = entry["source_row"]
        raw = upstream[source_row]
        prompt = raw["question"]
        if raw["id"] != entry["source_id"] or hashlib.sha256(prompt.encode()).hexdigest() != entry["prompt_sha256"]:
            raise ValueError(f"Source row {source_row} differs from Pub97")
        result.append(dict(question_id=entry["question_id"], source_row=source_row,
                           source_id=raw["id"], prompt=prompt, system_prompt=None))
    return result


def panel_ids(part, *, seed=DEFAULT_SEED, eval_size=DEFAULT_EVAL_SIZE):
    """List the source-ordered IDs for the selected part. Hash keys only determine membership."""
    if part not in ("discovery", "evaluation", "all"):
        raise ValueError("part must be discovery, evaluation, or all")
    ids = [r["question_id"] for r in panel_manifest()]
    if type(seed) is not int or type(eval_size) is not int or not 0 <= eval_size < len(ids):
        raise ValueError("Specify an integer seed and reserve between 0 and 96 evaluation questions")
    if part == "all":
        return ids

    def priority(question_id):
        key = f"pub97.split.v1|{seed}|{question_id}".encode("ascii")
        return hashlib.sha256(key).digest(), question_id

    evaluation = set(sorted(ids, key=priority)[:eval_size])
    return [qid for qid in ids if (qid in evaluation) == (part == "evaluation")]


def selected_layers(selection, layers):
    if (selection.get("panel") != "pub97" or selection.get("part") != "discovery"
            or selection.get("source_sha256") != SOURCE_SHA256 or "control_recipient" in selection):
        raise ValueError("Supply selected_neurons.json from Pub97 discovery")
    expected_ids = panel_ids("discovery", seed=selection["split_seed"], eval_size=selection["eval_size"])
    if selection["discovery_question_ids"] != expected_ids:
        raise ValueError("Discovery question IDs differ from the recorded Pub97 split")
    grouped = defaultdict(dict)
    for row in selection["selected"]:
        layer, neuron, gap = row["layer"], row["neuron"], row["mean_activation_gap"]
        if (type(layer) is not int or layer < 0 or type(neuron) is not int or neuron < 0
                or type(gap) not in (int, float) or not math.isfinite(gap) or neuron in grouped[layer]):
            raise ValueError("Invalid or duplicate discovery coordinate")
        grouped[layer][neuron] = row
    if not grouped:
        raise ValueError("Empty discovery population")
    if layers is None:
        layers = sorted(grouped)
    if len(set(layers)) != len(layers) or not set(layers) <= grouped.keys():
        raise ValueError("Each requested layer must have selected Pub97 coordinates")
    return {layer: [grouped[layer][i] for i in sorted(grouped[layer])] for layer in sorted(layers)}


def select_split(rows, part, *, seed=DEFAULT_SEED, eval_size=DEFAULT_EVAL_SIZE):
    """Select rows returned by load_pub97(), each question appearing exactly once."""
    chosen = panel_ids(part, seed=seed, eval_size=eval_size)
    manifest = panel_manifest()
    rows = list(rows)
    by_id = {row["question_id"]: row for row in rows}
    if len(rows) != len(manifest) or set(by_id) != {r["question_id"] for r in manifest}:
        raise ValueError("Expected a full Pub97 panel with one row per question")
    for entry in manifest:
        row = by_id[entry["question_id"]]
        if (row["source_id"] != entry["source_id"] or row["source_row"] != entry["source_row"]
                or row["system_prompt"] is not None
                or hashlib.sha256(row["prompt"].encode()).hexdigest() != entry["prompt_sha256"]):
            raise ValueError(f"Question binding mismatch: {entry['question_id']}")
    return [by_id[qid] for qid in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local upstream CSV at pinned revision")
    parser.add_argument("--part", choices=["discovery", "evaluation", "all"], required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--output", type=Path, required=True, help="New output JSONL file")
    args = parser.parse_args()
    rows = select_split(load_pub97(args.source), args.part, seed=args.seed, eval_size=args.eval_size)
    if args.part == "evaluation" and not rows:
        parser.error("Full-panel discovery reserves no questions for evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=f)
    print(f"Pub97 {args.part}, {len(rows)} questions, seed {args.seed}, evaluation size {args.eval_size}")


if __name__ == "__main__":
    main()
