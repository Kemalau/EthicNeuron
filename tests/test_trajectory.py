"""Numerical and file-contract tests for the Pub97 checkpoint trajectory."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge
import trajectory
from discovery import sha256
from pub97 import SOURCE_SHA256
from test_methods import tiny_model
from test_pipeline import TinyTokenizer


class TrajectoryTests(unittest.TestCase):
    def test_end_to_end_with_known_sigma_own_gaps_and_unsteered_judge_rows(self):
        model, tok = tiny_model(), TinyTokenizer()
        rows = [dict(prompt=text, system_prompt=None, question_id=qid, source_id=qid, source_row=i)
                for i, (qid, text) in enumerate([("a", "A"), ("b", "B?"), ("v", "V?")])]
        fixtures = [dict(question_id=row["question_id"], source_row=row["source_row"], source_id=row["source_id"],
                         prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest()) for row in rows]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ckpt").mkdir()
            (root / "ckpt/adapter_config.json").write_text("{}")
            (root / "ckpt/adapter_model.safetensors").write_text("weights fixture")
            gaps = np.arange(24, dtype=np.float64)[None, :] + 0.5
            np.savez(root / "all.npz", layers=[1], mean_gap=gaps)
            first = dict(layer=1, neuron=2, mean_activation_gap=gaps[0, 2])
            selection = dict(selected=[first], panel="pub97", part="discovery", split_seed=42, eval_size=1,
                             layers=[1], discovery_question_ids=["a", "b"], source_sha256=SOURCE_SHA256,
                             scores_sha256=sha256(root / "all.npz"), base="toy", base_revision="revision")
            (root / "selection.json").write_text(json.dumps(selection))
            manifest = dict(base="toy", base_revision="revision", calibration_question_ids=["a", "a", "b"],
                            checkpoints=[dict(seed=3, step=i, adapter="ckpt") for i in (12, 2)])
            (root / "manifest.json").write_text(json.dumps(manifest))
            def ids(part, **kwargs):
                return ["a", "b"] if part == "discovery" else ["v"]
            def part(_, stage, **kwargs):
                return rows[:2] if stage == "discovery" else rows[2:]
            with patch("trajectory.load_pub97", return_value=rows), patch("trajectory.select_split", side_effect=part), \
                 patch("pub97.panel_ids", side_effect=ids), \
                 patch("trajectory.load_model", return_value=(model, tok)), \
                 patch("judge.panel_ids", side_effect=ids), patch("judge.panel_manifest", return_value=fixtures):
                trajectory.main(["--source", "fixture", "--scores", str(root / "all.npz"),
                                 "--neurons", str(root / "selection.json"), "--checkpoints", str(root / "manifest.json"),
                                 "--output", str(root / "out"), "--device", "cpu", "--behavior-draws", "2",
                                 "--max-new-tokens", "2"])
                generated = judge.load_generations([root / "out/generations.jsonl"])
            result = json.loads((root / "out/trajectory.json").read_text())
            self.assertEqual(result["calibration_source"], "declared_training_prompts")
            self.assertEqual(result["calibration_token_count"], 7)
            self.assertEqual(result["monitor_question_ids"], ["v"])
            (control,) = result["random"]
            self.assertNotEqual(control["neuron"], 2)
            self.assertEqual(control["mean_activation_gap"], gaps[0, control["neuron"]])
            self.assertEqual([r["step"] for r in result["checkpoints"]], [0, 2, 12])
            self.assertEqual(result["checkpoints"][0]["selected"], 0)
            self.assertEqual(result["checkpoints"][1]["adapter_sha256"], sha256(root / "ckpt/adapter_model.safetensors"))
            self.assertEqual(len(generated), 6)
            self.assertTrue(all(r["alpha"] == 0 and not r["write_counts"] for r in generated))
            self.assertIsNone(generated[0]["adapter"])
            self.assertEqual(generated[0]["seed"], generated[2]["seed"])
            backbone, channels = trajectory.capture_setup(model, [first, control])
            with model.disable_adapter(), torch.no_grad():
                a = trajectory.capture_one(backbone, channels, [1, 9], boundary=False)
                b = trajectory.capture_one(backbone, channels, [1, 9, 10], boundary=False)
                base = trajectory.capture_one(backbone, channels, [1, 9, 10], boundary=True)
            expected_std = torch.cat([a, a, b]).std(0, correction=0).clamp_min(.001)
            torch.testing.assert_close(torch.tensor(result["sigma"], dtype=torch.float64), expected_std)
            with torch.no_grad():
                current = trajectory.capture_one(backbone, channels, [1, 9, 10], boundary=True)
            expected = (current - base) / expected_std
            self.assertAlmostEqual(result["checkpoints"][1]["selected"], expected[0, 0].item())
            self.assertAlmostEqual(result["checkpoints"][1]["random"], expected[0, 1].item())

    def test_manifest_rejects_duplicate_steps_mutable_remote_and_incompatible_base(self):
        with TemporaryDirectory() as directory:
            p = Path(directory) / "m.json"
            base = dict(base="base", base_revision="f" * 40)
            population = dict(base, panel="pub97", split_seed=42, eval_size=20, source_sha256=SOURCE_SHA256,
                              discovery_question_ids=["q"])
            row = dict(seed=1, step=9, adapter="owner/checkpoint", adapter_revision="a" * 40)
            for rows, change, error in [([row, row], {}, "Duplicate"),
                                         ([dict(row, adapter_revision="main")], {}, "immutable"),
                                         ([row], dict(base="other"), "base differs"),
                                         ([row], dict(split_seed=18), "provenance differs"),
                                         ([row], dict(selection_sha256="other"), "provenance differs"),
                                         ([row], dict(seed=3), "seed differs"),
                                         ([dict(row, arm="sft"), dict(row, seed=2, arm="selected")],
                                          {}, "separate manifests")]:
                p.write_text(json.dumps(dict(base, **change) | dict(checkpoints=rows)))
                with self.assertRaisesRegex(ValueError, error):
                    trajectory.read_checkpoints(p, population, "bound")
