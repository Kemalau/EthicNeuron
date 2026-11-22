"""CPU checks for per-layer write controls and centred mixing directions."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.vocabulary_controls import centered_fastica, dose_weighted_writes, matched_random_writes


class RandomWriteTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(27)
        self.weights = torch.from_numpy(rng.normal(size=(2, 3, 28)).astype(np.float32))
        self.gaps = rng.normal(size=(2, 28))
        self.chosen = [1, 9, 20]
        self.rows = [
            dict(layer=layer, neuron=i, mean_activation_gap=self.gaps[layer, i])
            for layer in (1, 0) for i in reversed(self.chosen)
        ]

    def generate(self, **kwargs):
        return matched_random_writes(self.rows, [0, 1], self.gaps,
                                     lambda layer, ids: self.weights[layer][:, ids], **kwargs)

    def test_layer_draw_dose_and_norm(self):
        outputs = list(self.generate())
        self.assertEqual(len(outputs), 20)
        pool = sorted(set(range(28)).difference(self.chosen))
        for idx, (audit, writes) in enumerate(outputs):
            layer, seed = divmod(idx, 10)
            self.assertEqual((audit["layer"], audit["seed"]), (layer, seed))
            members = sorted(np.random.default_rng(seed).choice(pool, 3, replace=False).tolist())
            self.assertEqual(audit["neurons"], members)
            w = self.weights[layer].double()
            gaps = torch.from_numpy(self.gaps[layer])
            raw = (w[:, members] * (2 * gaps[members])).T
            target = (w[:, self.chosen] @ (2 * gaps[self.chosen])).norm().item()
            expected = raw * (target / raw.sum(0).norm())
            torch.testing.assert_close(writes.double(), expected, rtol=2e-6, atol=1e-7)
            self.assertAlmostEqual(audit["matched_write_norm"] / target, 1., places=6)
            np.testing.assert_array_equal(audit["mean_gaps"], self.gaps[layer, members])

    def test_mismatched_or_undefined_writes(self):
        with self.assertRaisesRegex(ValueError, "Undefined selected"):
            list(self.generate(alpha=0))
        self.weights[:, :, self.chosen] = 1.
        self.weights[:, :, sorted(set(range(28)).difference(self.chosen))] = 0.
        with self.assertRaisesRegex(ValueError, "Undefined random"):
            list(self.generate())
        self.rows[0]["mean_activation_gap"] += 1.
        with self.assertRaisesRegex(ValueError, "Selected gap differs"):
            list(self.generate())

    def test_weight_layout_and_input_precision(self):
        cols = torch.tensor([[1., 4.], [2., -2.], [7., 0.]], dtype=torch.bfloat16)
        actual = dose_weighted_writes(cols, [1., -2.], 2.)
        torch.testing.assert_close(actual, torch.tensor([[2., 4., 14.], [-16., 8., 0.]]))


class MixingTests(unittest.TestCase):
    def setUp(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest("scikit-learn is an optional FastICA dependency")
        rng = np.random.default_rng(8)
        source = rng.laplace(size=(150, 3))
        self.writes = torch.tensor(source @ rng.normal(size=(3, 5)) + 3., dtype=torch.float32)

    def test_centering_mixing_columns_and_energy(self):
        from sklearn.decomposition import FastICA
        dirs, audit = centered_fastica(self.writes, 2, seed=1)
        matrix = self.writes.numpy().astype(np.float64)
        centered = matrix - matrix.mean(axis=0)
        sv = np.linalg.svd(centered, compute_uv=False)
        self.assertEqual(audit["centered_rank_fp32"], 3)
        self.assertAlmostEqual(audit["explained_centered_fraction"], sum(sv[:2]**2)/sum(sv**2), places=10)
        self.assertTrue(np.all(dirs.numpy() @ matrix.sum(axis=0) >= 0))
        ref = FastICA(n_components=2, whiten="unit-variance", whiten_solver="svd", random_state=1,
                      max_iter=4000, tol=1e-4).fit(matrix)
        np.testing.assert_allclose(np.abs(dirs.numpy()), np.abs(ref.mixing_.T), rtol=2e-6)
        self.assertEqual(dirs.shape, (2, 5))
        self.assertIsInstance(audit["converged"], bool)

    def test_centered_rank_limit(self):
        with self.assertRaisesRegex(ValueError, "rank 3"):
            centered_fastica(self.writes, 4)
        with self.assertRaisesRegex(ValueError, "rank 0"):
            centered_fastica(torch.ones((5, 2)), 1)


if __name__ == "__main__":
    unittest.main()
