"""Check ties and Unicode filtering at both ends of the output head."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.vocabulary_ranks import control_rank_rows, ranked_tokens


class Tokenizer:
    all_special_ids = [3]
    pieces = ["+!", " Foo", "切", "<special>", " Ⅻ", "foO", "\t\n", "\u0301"]

    def __init__(self):
        self.decoded = []

    def convert_ids_to_tokens(self, i):
        return self.pieces[i] if i < len(self.pieces) else None

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        self.decoded.append(ids[0])
        return self.pieces[ids[0]]


class VocabularyRankTests(unittest.TestCase):
    def test_both_ends_preserve_ids_signed_scores_and_competition_ranks(self):
        # The head includes an extra output row at 8, beyond the tokenizer.
        z = torch.tensor([5., 4., -2., -2., -2., 4., 3., -1., 5.], dtype=torch.bfloat16)
        rows = list(ranked_tokens(z, Tokenizer(), 5, 3))
        def table(view, end):
            return [(r["position"], r["raw_rank"], r["token_id"], r["token"], r["score"])
                    for r in rows if (r["view"], r["end"]) == (view, end)]
        self.assertEqual(table("raw", "positive"), [
            (1, 1, 0, "+!", 5.), (2, 1, 8, "", 5.), (3, 3, 1, " Foo", 4.),
            (4, 3, 5, "foO", 4.), (5, 5, 6, "\t\n", 3.)])
        self.assertEqual(table("filtered", "negative"), [
            (1, 1, 2, "切", -2.), (2, 1, 4, " Ⅻ", -2.), (3, 6, 1, " Foo", 4.)])
        self.assertEqual(table("filtered", "positive"), [
            (1, 3, 1, " Foo", 4.), (2, 3, 5, "foO", 4.), (3, 7, 2, "切", -2.)])
        self.assertEqual(set(rows[0]), {"view", "end", "position", "raw_rank", "token_id", "token", "score"})

    def test_fetch_only_needed_ends(self):
        tok = Tokenizer()
        rows = list(ranked_tokens(np.arange(8, dtype=np.float32), tok, 1, 0))
        self.assertEqual(tok.decoded, [7, 0])
        self.assertEqual([r["view"] for r in rows], ["raw", "raw"])

    def test_special_added_token_outside_all_special_ids(self):
        tok = Tokenizer()
        tok.added_tokens_decoder = {1: SimpleNamespace(special=True), 2: SimpleNamespace(special=False)}
        rows = list(ranked_tokens([1., 10., 2.], tok, 2, 1))
        positive = [(r["view"], r["token_id"], r["raw_rank"]) for r in rows if r["end"] == "positive"]
        self.assertEqual(positive, [("raw", 1, 1), ("raw", 2, 2), ("filtered", 2, 2)])

    def test_smaller_vocab_and_insufficient_retained(self):
        rows = list(ranked_tokens(np.array([-4, -4, -4]), Tokenizer(), 100, 100))
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["raw_rank"] for r in rows}, {1})
        self.assertEqual([r["token_id"] for r in rows if r["end"] == "negative"
                          and r["view"] == "filtered"], [1, 2])

    def test_bad_scores(self):
        with self.assertRaisesRegex(ValueError, "finite vocabulary scores"):
            list(ranked_tokens([1., np.nan], Tokenizer(), 1, 0))

    def test_same_end_controls_share_rank_at_top50_cutoff(self):
        # Select one token on both ends, repeated across the raw and filtered views.
        selected = [r for r in ranked_tokens(np.arange(55), Tokenizer(), 55, 3)
                    if r["token_id"] == 1]
        self.assertTrue(any(r["view"] == "filtered" for r in selected))
        random = {}
        for seed in range(10):
            # At the positive end token 1 ties three others at 50 or two at 51.
            # Two scores are strictly lower for every seed.
            values = np.zeros(55)
            values[[0, 2]] = -1
            values[-(49 if seed < 3 else 50):] = 1
            random[seed] = values
        pos, neg = control_rank_rows(selected, random)
        self.assertEqual((pos["end"], pos["token_id"], pos["token"], pos["selected_rank"]),
                         ("positive", 1, " Foo", 54))
        self.assertEqual([pos[f"seed_{i}_rank"] for i in range(10)], [50] * 3 + [51] * 7)
        self.assertEqual((pos["best_random_rank"], pos["random_top50"]), (50, 3))
        self.assertEqual((neg["end"], neg["selected_rank"], neg["best_random_rank"], neg["random_top50"]),
                         ("negative", 2, 3, 10))
        self.assertTrue(all(neg[f"seed_{i}_rank"] == 3 for i in range(10)))

    def test_controls_require_ten_same_size_vocabularies(self):
        selected = list(ranked_tokens([1., 2.], Tokenizer(), 1, 0))
        controls = {i: np.array([1., 2.]) for i in range(9)}
        with self.assertRaisesRegex(ValueError, "per seed"):
            list(control_rank_rows(selected, controls))
        controls[9] = np.ones(3)
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            list(control_rank_rows(selected, controls))


if __name__ == "__main__":
    unittest.main()
