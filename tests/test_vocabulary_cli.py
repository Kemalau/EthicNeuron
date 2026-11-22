"""Trace a Pub97 selection through the vocabulary CLI using small CPU tensors."""

from contextlib import ExitStack
import csv
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from pub97 import SOURCE_SHA256, panel_ids
import vocabulary_readout


class Tokenizer:
    all_special_ids = [0]
    pieces = ["<s>", "!", " A", "b", "字", " IV", "\n", "z"] + [f"w{i}" for i in range(8, 80)]

    def decode(self, ids, **kwargs):
        return self.pieces[ids[0]]


class ReadoutWorkflowTests(unittest.TestCase):
    def test_layer_joint_and_repeated_matched_draws(self):
        rng = np.random.default_rng(1)
        cols = rng.normal(size=(2, 3, 8)).astype(np.float32)
        gaps = rng.normal(size=(2, 8))
        chosen = [[1, 5], [2, 3]]
        maps = {layer: torch.from_numpy(rng.normal(size=(3, 3)).astype(np.float32)) for layer in range(2)}
        head = torch.nn.Linear(3, len(Tokenizer.pieces), bias=False)
        head.weight.data.copy_(torch.from_numpy(rng.normal(size=head.weight.shape).astype(np.float32)))
        norm = torch.nn.RMSNorm(3, eps=1e-3)
        norm.weight.data.copy_(torch.tensor([1., 0.5, 1.75]))
        layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(8, 3))) for _ in range(2)]
        core = SimpleNamespace(model=SimpleNamespace(layers=layers), config=SimpleNamespace(model_type="qwen2"))
        model = SimpleNamespace(get_base_model=lambda: core)

        # Compute the expected scores in FP64 independently of the package's
        # write, transport, normalization, ranking and control implementations.
        def write(layer, ids):
            return cols[layer][:, ids].astype(np.float64) @ (2 * gaps[layer, ids])

        def decode(vector):
            rms = np.sqrt(np.mean(vector**2) + 1e-3)
            return head.weight.detach().numpy() @ (norm.weight.detach().numpy() * vector / rms)

        target = [write(layer, chosen[layer]) for layer in range(2)]
        selected = [dict(layer=layer, neuron=i, mean_activation_gap=float(gaps[layer, i]),
                         rank=k, score=1/k, mean_contribution=1/k, positive_fraction=1.)
                    for k, (layer, i) in enumerate([(1, 3), (0, 1), (1, 2), (0, 5)], 1)]
        with TemporaryDirectory() as tmp, ExitStack() as mocks:
            folder = Path(tmp)
            scores_file = folder / "all_scores.npz"
            np.savez(scores_file, mean_gap=gaps, layers=[0, 1])
            selection_file = folder / "selected_neurons.json"
            selection_file.write_text(json.dumps(dict(
                panel="pub97", part="discovery", eval_size=20, split_seed=42, layers=[0, 1],
                base="toy-base", adapter="toy-adapter", base_revision="a" * 40, adapter_revision="b" * 40,
                judge_model="fixture", rubric_sha256=hashlib.sha256(
                    (PACKAGE / "rubrics/alignment_coherence_rubrics.json").read_bytes()).hexdigest(),
                source_sha256=SOURCE_SHA256, input_sha256="a" * 64, selected=selected,
                scores_sha256=hashlib.sha256(scores_file.read_bytes()).hexdigest(),
                discovery_question_ids=panel_ids("discovery"), positive_candidates=16, scored_coordinates=16,
                paired_questions=2, position="final assistant-header token")))
            lens = folder / "fixture.pt"
            lens.write_bytes(b"CPU test map supplied to the mocked loader")
            for name, value in [("load_readout_model", (model, Tokenizer())), ("readout_modules", (norm, head)),
                                ("load_transport", (maps, {"kind": "jacobian", "n_prompts": 3}))]:
                mocks.enter_context(patch.object(vocabulary_readout, name, return_value=value))
            mocks.enter_context(patch.object(vocabulary_readout, "write_columns",
                side_effect=lambda model, layer, neurons, write_weights: torch.from_numpy(cols[layer][:, neurons])))
            common = ["--neurons", str(selection_file), "--layers", "0", "1", "--device", "cpu",
                      "--top-k", "3", "--filtered-top-k", "2", "--per-neuron"]
            out = folder / "transported"
            ica = ["--ica-components", "1", "--ica-seeds", "0"] if find_spec("sklearn") else []
            vocabulary_readout.main(common + ica + ["--scores", str(scores_file), "--method", "jacobian",
                                                    "--lens", str(lens), "--output", str(out)])
            plain = folder / "plain"
            vocabulary_readout.main(common + ["--method", "logit", "--output", str(plain)])
            with np.load(out / "scores.npz") as result, np.load(plain / "scores.npz") as direct:
                def check(name, vector):
                    np.testing.assert_allclose(result[name], decode(vector), rtol=2e-5, atol=3e-6)

                transformed = [maps[l].numpy() @ target[l] for l in range(2)]
                for layer in range(2):
                    check(f"selected_layer_{layer}_jacobian", transformed[layer])
                    for neuron in chosen[layer]:
                        check(f"neuron_{layer}_{neuron}_jacobian", maps[layer].numpy() @ write(layer, [neuron]))
                check("selected_joint_jacobian", sum(transformed))
                check("selected_joint_logit", sum(target))
                np.testing.assert_array_equal(direct["selected_joint_logit"], result["selected_joint_logit"])
                if ica:
                    writes = np.stack([maps[l].numpy() @ write(l, [n]) for l in range(2) for n in chosen[l]])
                    _, singular, vh = np.linalg.svd(writes - writes.mean(axis=0), full_matrices=False)
                    # Unit-variance one-component whitening yields unit population variance
                    # of the scores; the mixing column carries the remaining scale.
                    component = vh[0] * (singular[0] / np.sqrt(len(writes)))
                    if component @ writes.sum(axis=0) < 0:
                        component = -component
                    check("ica_1_seed_0_component_0_jacobian", component)
                joint_scores = []
                for seed in range(10):
                    joint = np.zeros(3)
                    for layer in range(2):
                        pool = sorted(set(range(8)) - set(chosen[layer]))
                        sampled = sorted(np.random.default_rng(seed).choice(pool, 2, replace=False))
                        raw = write(layer, sampled)
                        local = raw * (np.linalg.norm(target[layer]) / np.linalg.norm(raw))
                        carried = maps[layer].numpy() @ local
                        check(f"random_{seed}_layer_{layer}_jacobian", carried)
                        joint += carried
                    check(f"random_{seed}_joint_jacobian", joint)
                    joint_scores.append(decode(joint))
                expected = result["selected_joint_jacobian"]
                with (out / "tokens.csv").open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(set(reader.fieldnames), {"group", "view", "end", "position", "raw_rank",
                                                             "token_id", "token", "score"})
                    rows = [row for row in reader if row["group"] == "selected_joint_jacobian"]
                self.assertEqual(len(rows), 10)
                for row in rows:
                    i = int(row["token_id"])
                    oriented = expected if row["end"] == "positive" else -expected
                    ids = np.argsort(-oriented, kind="stable").tolist()
                    if row["view"] == "filtered":
                        ids = [j for j in ids if j not in (0, 1, 6)]
                    self.assertEqual(i, ids[int(row["position"]) - 1])
                    self.assertEqual(int(row["raw_rank"]), 1 + int(np.sum(oriented > oriented[i])))
                    self.assertEqual(row["token"], Tokenizer.pieces[i])
                    self.assertEqual(float(row["score"]), float(expected[i]))

                with (out / "control_ranks.csv").open(encoding="utf-8", newline="") as stream:
                    controls = [row for row in csv.DictReader(stream) if row["group"] == "selected_joint_jacobian"]
                displayed = {(r["end"], r["token_id"]): int(r["raw_rank"]) for r in rows}
                self.assertEqual(len(controls), len(displayed))
                self.assertEqual({(r["end"], r["token_id"]) for r in controls}, set(displayed))
                counts = []
                for row in controls:
                    token_id = int(row["token_id"])
                    sign = 1 if row["end"] == "positive" else -1
                    ranks = [1 + np.count_nonzero(sign * z > sign * z[token_id]) for z in joint_scores]
                    self.assertEqual([int(row[f"seed_{s}_rank"]) for s in range(10)], ranks)
                    self.assertEqual(int(row["selected_rank"]), displayed[row["end"], row["token_id"]])
                    self.assertEqual(int(row["best_random_rank"]), min(ranks))
                    hits = sum(rank <= 50 for rank in ranks)
                    self.assertEqual(int(row["random_top50"]), hits)
                    counts.append(hits)
                # Fixture exercises random ranks on both sides of the 50 cutoff.
                self.assertTrue(any(0 < count < 10 for count in counts))

    def test_wrong_panel_rejected_before_model_or_output(self):
        with TemporaryDirectory() as tmp, patch.object(vocabulary_readout, "load_readout_model") as load:
            folder = Path(tmp)
            selection = folder / "selection.json"
            selection.write_text(json.dumps({"panel": "unbound", "part": "discovery"}))
            output = folder / "absent"
            with self.assertRaisesRegex(ValueError, "Pub97"):
                vocabulary_readout.main(["--neurons", str(selection), "--method", "logit", "--output", str(output)])
            load.assert_not_called()
            self.assertEqual(list(folder.iterdir()), [selection])


if __name__ == "__main__":
    unittest.main()
