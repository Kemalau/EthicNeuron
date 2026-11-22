"""Exercise paper ranking ablation and K sweeps from frozen aggregates on CPU."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discovery import rank_statistics, sha256
from pub97 import SOURCE_SHA256, panel_ids
from ranking_variants import export_prefixes, ranking_prefixes


class RankingVariantTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scores = self.root / "all_scores.npz"
        self.neurons = self.root / "selected_neurons.json"
        mean = np.array([[4., 3., -1.], [4., 2., 0.]])
        freq = np.array([[.5, 1., .0], [1., 1., .0]])
        gap = np.array([[-.75, 1., 2.], [-1., 0., -2.]])
        layers = [0, 2]
        rows, score, count = rank_statistics(mean, freq, gap, layers, 2)
        self.arrays = dict(mean_contribution=mean, positive_fraction=freq, mean_gap=gap,
                           layers=layers, score=score)
        np.savez_compressed(self.scores, **self.arrays)
        ids = panel_ids("discovery", seed=11, eval_size=2)
        self.selection = dict(selected=rows, panel="pub97", part="discovery", layers=layers,
                              split_seed=11, eval_size=2, discovery_question_ids=ids, paired_questions=2,
                              paired_question_ids=ids[:2], source_sha256=SOURCE_SHA256,
                              scores_sha256=sha256(self.scores), input_sha256="a"*64,
                              base="base-fixture", adapter="adapter-fixture",
                              scored_coordinates=score.size, positive_candidates=count)
        self.neurons.write_text(json.dumps(self.selection))

    def test_weighted_extends_original_with_immutable_gaps_and_binding(self):
        output = self.root / "out"
        export_prefixes(self.neurons, self.scores, output, [4, 1], "weighted")
        small, large = [json.loads((output / f"top{k}.json").read_text()) for k in (1, 4)]
        self.assertEqual(small["selected"], large["selected"][:1])
        self.assertEqual(large["selected"][:2], self.selection["selected"])
        self.assertEqual([(r["layer"],r["neuron"]) for r in large["selected"]],
                         [(2, 0), (0, 1), (2, 1), (0, 0)])
        self.assertEqual(large["selected"][-1]["mean_activation_gap"], -.75)
        self.assertEqual(large["parent_selection_sha256"], sha256(self.neurons))
        for key in ("scores_sha256", "input_sha256", "paired_question_ids", "base", "adapter"):
            self.assertEqual(large[key], self.selection[key])

    def test_unweighted_ties_use_layer_then_channel_with_same_pool(self):
        small, large = ranking_prefixes(self.selection, self.arrays, [2, 4], "unweighted")
        self.assertEqual(small["selected"], large["selected"][:2])
        self.assertEqual([(r["layer"], r["neuron"], r["score"]) for r in large["selected"]],
                         [(0, 0, 4.), (2, 0, 4.), (0, 1, 3.), (2, 1, 2.)])
        self.assertEqual([r["mean_activation_gap"] for r in large["selected"]], [-.75, -1., 1., 0.])
        self.assertEqual(large["ranking_weighting"], "unweighted")

    def test_source_and_archive_validation(self):
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "positive candidates"):
            export_prefixes(self.neurons, self.scores, output, [1, 6], "weighted")
        self.assertFalse(output.exists())
        original = self.selection["selected"]
        with self.assertRaisesRegex(ValueError, "Frozen scores"):
            list(ranking_prefixes(dict(self.selection, selected=original[::-1]), self.arrays, [1], "unweighted"))
        np.savez(self.scores, **self.arrays, extra=0)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            export_prefixes(self.neurons, self.scores, output, [1], "weighted")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
