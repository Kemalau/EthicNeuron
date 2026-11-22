"""Export nested Pub97 steering populations from one frozen discovery score archive."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from discovery import rank_statistics, sha256
from pub97 import SOURCE_SHA256, panel_ids


def ranking_prefixes(source, archive, top_ks, weighting):
    """Verify the original weighted ranking and freeze prefixes under one scoring rule."""
    if (source.get("panel") != "pub97" or source.get("part") != "discovery"
            or source.get("source_sha256") != SOURCE_SHA256
            or source.get("ranking_weighting", "weighted") != "weighted"
            or "parent_selection_sha256" in source or "control_recipient" in source):
        raise ValueError("Expected the original weighted Pub97 discovery selection")
    expected_ids = panel_ids("discovery", seed=source["split_seed"], eval_size=source["eval_size"])
    if source["discovery_question_ids"] != expected_ids:
        raise ValueError("Discovery question IDs differ from the recorded Pub97 split")
    if "paired_question_ids" in source:
        paired = source["paired_question_ids"]
        if (len(paired) != len(set(paired)) or len(paired) != source["paired_questions"]
                or not set(paired) <= set(expected_ids)):
            raise ValueError("Invalid retained paired question IDs")

    if (not top_ks or any(type(k) is not int or k < 1 for k in top_ks)
            or len(set(top_ks)) != len(top_ks)):
        raise ValueError("Specify distinct positive prefix lengths; use zero dose for the baseline")
    if weighting not in ("weighted", "unweighted"):
        raise ValueError("Choose weighted or unweighted ranking")
    layers = np.asarray(archive["layers"])
    if layers.ndim != 1 or layers.dtype.kind not in "iu" or layers.tolist() != source["layers"]:
        raise ValueError("Discovery layers differ between ranking and archive")
    stats = (archive["mean_contribution"], archive["positive_fraction"], archive["mean_gap"], layers.tolist())
    original = source["selected"]
    weighted_k = max(len(original), max(top_ks)) if weighting == "weighted" else len(original)
    ranked, scores, positives = rank_statistics(*stats, weighted_k)
    if (not original or not np.array_equal(scores, archive["score"])
            or ranked[:len(original)] != original
            or scores.size != source["scored_coordinates"] or positives != source["positive_candidates"]):
        raise ValueError("Frozen scores or selected channels differ from the discovery ranking")
    if weighting == "unweighted":
        ranked, _, positives = rank_statistics(*stats, max(top_ks), weighting="unweighted")

    for k in sorted(top_ks):
        yield dict(source, selected=ranked[:k], ranking_weighting=weighting,
                   prefix_length=k, positive_candidates=positives)


def export_prefixes(neurons, scores, output_dir, top_ks, weighting):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    payload = neurons.read_bytes()
    source = json.loads(payload)
    if source["scores_sha256"] != sha256(scores):
        raise ValueError("Score archive SHA-256 differs from this discovery selection")
    parent_sha = hashlib.sha256(payload).hexdigest()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with np.load(scores, allow_pickle=False) as archive, TemporaryDirectory(dir=output_dir.parent) as tmp:
        new = Path(tmp) / "prefixes"
        new.mkdir()
        for selection in ranking_prefixes(source, archive, top_ks, weighting):
            selection["parent_selection_sha256"] = parent_sha
            path = new / f"top{selection['prefix_length']}.json"
            path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        new.rename(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=Path, required=True, help="Original Pub97 selected_neurons.json")
    parser.add_argument("--scores", type=Path, required=True, help="Matching all_scores.npz")
    parser.add_argument("--weighting", choices=["weighted", "unweighted"], default="weighted")
    parser.add_argument("--top-k", type=int, nargs="+", required=True, help="One or more positive prefix lengths")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for topK.json files")
    args = parser.parse_args()
    export_prefixes(args.neurons, args.scores, args.output_dir, args.top_k, args.weighting)


if __name__ == "__main__":
    main()
