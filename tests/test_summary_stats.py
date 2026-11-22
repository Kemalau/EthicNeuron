"""Statistics on a fixed response denominator and independent training seeds."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import judge
from methods.summary_stats import training_seed_summary, wilson95


class ResponseIntervalTests(unittest.TestCase):
    def test_wilson_limits_and_extremes(self):
        lo, hi = wilson95(5, 10)
        self.assertAlmostEqual(lo, .236593090512564)
        self.assertAlmostEqual(hi, .763406909487436)
        self.assertEqual(wilson95(0, 10)[0], 0)
        self.assertAlmostEqual(wilson95(0, 10)[1], .277532799862889)
        self.assertEqual(wilson95(10, 10)[1], 1)
        for k, n in [(-1, 10), (11, 10), (0, 0), (1.5, 3), (True, 5)]:
            with self.subTest(k=k, n=n), self.assertRaises(ValueError):
                wilson95(k, n)

    def test_judge_counts_include_unknowns_and_truncated_draws(self):
        rows = [dict(condition="baseline", part="evaluation", em=em, low_coherence=low,
                     truncated=em is None, axes={"aligned": {"score": None if em is None else 20}})
                for em, low in [(True, False), (False, True), (None, None), (None, None)]]
        result, = judge.summarize(rows)
        self.assertEqual((result["n"], result["em"], result["unknown_em"], result["truncated"]), (4, 1, 2, 2))
        for name in ("em", "low_coherence"):
            self.assertEqual(result[name + "_pct"], 25)
            self.assertAlmostEqual(result[name + "_wilson95_low_pct"], 4.558726080973)
            self.assertAlmostEqual(result[name + "_wilson95_high_pct"], 69.93581574176)


class IndependentSeedTests(unittest.TestCase):
    def test_equal_seed_weight_and_sample_sd(self):
        # Three independently trained checkpoints with unequal response counts.
        result = training_seed_summary({2: 3/4, 0: 0/1, 1: 3/4})
        self.assertEqual(result["training_seeds"], [0, 1, 2])
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual(result["mean"], .5)
        self.assertAlmostEqual(result["sample_sd"], .4330127018922193)
        self.assertEqual((result["minimum"], result["maximum"]), (0., .75))

    def test_reject_draws_missing_seeds_and_nonfinite_values(self):
        for outcomes in ([1., 3.], {0: 1.}, {"seed": 1, 0: 0}, {0: float("nan"), 2: 0}):
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                training_seed_summary(outcomes)


if __name__ == "__main__":
    unittest.main()
