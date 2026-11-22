"""Offline tests with a native Qwen2 model and a nonzero LoRA update."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_controls import build_control, main
from discovery import decoder_layers, sha256
from steering import SparseSteering
from test_methods import tiny_model


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.layers = decoder_layers(self.model)
        self.width = self.layers[0].mlp.down_proj.in_features
        self.gaps = np.random.default_rng(117).normal(size=(2,self.width))
        self.selected = [(0,0), (0,3), (1,9)]
        self.ranking = {"base": "toy", "adapter": "toy-med", "layers": [0,1],
                        "selected": [dict(layer=l, neuron=i, mean_activation_gap=self.gaps[l,i])
                                     for l,i in self.selected]}

    def test_receiver_scaling_and_cli(self):
        groups = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root/"scores.npz", mean_gap=self.gaps, layers=[0,1])
            self.ranking["scores_sha256"] = sha256(root/"scores.npz")
            (root/"selection.json").write_text(json.dumps(self.ranking))
            for receiver in ["base","medical"]:
                output = root/(receiver+".json")
                args = ["matched_controls", "--neurons", str(root/"selection.json"), "--scores",
                        str(root/"scores.npz"), "--output", str(output), "--recipient", receiver,
                        "--seed", "72", "--device", "cpu", "--local-files-only"]
                with patch.object(sys, "argv", args), patch("matched_controls.load_model",
                                                           return_value=(self.model, None)):
                    main()
                result = json.loads(output.read_text())
                group = result["selected"]
                groups.append(group)
                self.assertEqual(result["control_recipient"],receiver)
                self.assertEqual(len(result["selection_sha256"]),64)
                ids = [(r["layer"],r["neuron"]) for r in group]
                self.assertFalse(set(ids) & set(self.selected))
                self.assertEqual(Counter(l for l,_ in ids),Counter(l for l,_ in self.selected))
                for layer in range(2):
                    module = self.layers[layer].mlp.down_proj
                    w = module.base_layer.weight
                    if receiver == "medical":
                        w = w + module.get_delta_weight("default")
                    ic = [r["neuron"] for r in group if r["layer"] == layer]
                    it = [i for l,i in self.selected if l == layer]
                    target = w[:,it] @ torch.tensor(self.gaps[layer,it],dtype=torch.float32)
                    gap = torch.tensor([r["mean_activation_gap"] for r in group if r["layer"] == layer])
                    torch.testing.assert_close((w[:,ic]@gap).norm(), target.norm(), rtol=3e-5, atol=1e-6)
                with SparseSteering(self.layers, group, 1, torch.ones(1, 2)) as edit:
                    self.model(input_ids=torch.tensor([[1,2]]))
                self.assertEqual(edit.edited_calls, {0:1,1:1})
        self.assertEqual([r["neuron"] for r in groups[0]],[r["neuron"] for r in groups[1]])
        self.assertNotEqual(groups[0],groups[1])

    def test_gap_mismatch_and_impossible_scale(self):
        gaps = self.gaps.copy()
        gaps[0,0] += 1
        with self.assertRaisesRegex(ValueError,"does not match"):
            build_control(self.model, self.ranking, [0,1], gaps, "base", 72)
        gaps.fill(0)
        for l,i in self.selected:
            gaps[l,i] = self.gaps[l,i]
        with self.assertRaisesRegex(ValueError,"Undefined"):
            build_control(self.model, self.ranking, [0,1], gaps, "base", 72)


if __name__ == "__main__":
    unittest.main()
