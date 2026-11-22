"""Offline checks of Pub97 two-order scope, decoding and blind extraction."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import choice_evaluation as cli
from generate import run_batch
from methods.choices import resume_presentations
from pub97 import SOURCE_SHA256, panel_ids
from test_methods import tiny_model
from test_pipeline import TinyTokenizer


class ChoiceWorkflowTests(unittest.TestCase):
    def test_verified_rows_and_default_split(self):
        source = os.environ.get("PUB97_SOURCE")
        if not source:
            self.skipTest("Set PUB97_SOURCE for source-specific coverage")
        with self.assertRaisesRegex(ValueError, "no eligible"):
            cli.tasks_for(source, 42, 20)
        tasks = cli.tasks_for(source, 0, 20)
        self.assertEqual([(t["question_id"], t["order"]) for t in tasks],
                         [("extended_004", "original"), ("extended_004", "swapped")])
        q = tasks[0]["prompt"]
        a, b = [line[4:] for line in q.splitlines() if line.startswith(("(A) ", "(B) "))]
        self.assertEqual(tasks[1]["prompt"], q.replace("(A) " + a, "(A) " + b).replace("(B) " + b, "(B) " + a))

    def test_greedy_generation_uses_single_beam(self):
        model = tiny_model()
        tasks = [dict(prompt="Which?", system_prompt=None)]
        with patch.object(model, "generate", side_effect=lambda input_ids, **kw:
                          torch.cat((input_ids, torch.tensor([[2]])), dim=1)) as generate:
            rows = run_batch(model, TinyTokenizer(), [], tasks, "base", 0., 0, 256, greedy=True)
        opts = generate.call_args.kwargs
        self.assertEqual((opts["do_sample"], opts["num_beams"], opts["max_new_tokens"]), (False, 1, 256))
        self.assertTrue(rows[0]["ended_eos"])

    def test_all_arms_extraction_blinding_and_failed_denominator(self):
        q = dict(question_id="extended_004", source_id="resume_ranking_4", source_row=4, system_prompt=None,
                 prompt="Choose A or B.\n(A) First person.\n(B) Second person.\nGive a reason.")
        entry = dict(q, prompt_sha256=hashlib.sha256(q["prompt"].encode()).hexdigest())
        with TemporaryDirectory() as td, ExitStack() as patches:
            root = Path(td)
            neurons, generations, judged = [root / name for name in ("neurons.json", "answers.json", "judged.json")]
            selection = dict(panel="pub97", part="discovery", source_sha256=SOURCE_SHA256, split_seed=0, eval_size=20,
                             discovery_question_ids=panel_ids("discovery", seed=0),
                             selected=[dict(layer=0, neuron=0, mean_activation_gap=.3)],
                             base="fake-base", adapter="fake-adapter", base_revision="b", adapter_revision="a")
            neurons.write_text(json.dumps(selection))
            patches.enter_context(patch.object(cli, "load_pub97", return_value=[q]))
            patches.enter_context(patch.object(cli, "select_split", side_effect=lambda rows, *a, **kw: rows))
            patches.enter_context(patch("methods.choices.panel_manifest", return_value=[entry]))
            patches.enter_context(patch.object(cli, "load_model", return_value=(None, None)))
            def native(model, tok, selected, tasks, recipient, alpha, seed, cap, *, greedy):
                self.assertTrue(greedy)
                self.assertEqual(cap, 256)
                return [dict(t, recipient=recipient, alpha=alpha, response="Fixture answer.",
                             truncated=t["order"] == "swapped") for t in tasks]
            patches.enter_context(patch.object(cli, "run_batch", side_effect=native))
            self.assertEqual(cli.main(["generate", "--source", "fixture", "--neurons", str(neurons),
                                       "--device", "cpu", "--output", str(generations)]), 0)
            request = Mock(side_effect=["A", "B", "A", "A", cli.ServiceError("HTTP 503"), "Ambiguous"])
            patches.enter_context(patch.object(cli, "transport", return_value=request))
            args = ["extract", "--source", "fixture", "--input", str(generations), "--endpoint",
                    "https://fixture.invalid/chat", "--model", "independent", "--output", str(judged)]
            with self.assertRaisesRegex(ValueError, "credential-free"):
                cli.main([s + "?key=fixture" if s.startswith("https:") else s for s in args])
            self.assertFalse(judged.exists())
            request.assert_not_called()
            self.assertEqual(cli.main(args), 1)
            data = json.loads(judged.read_text())
            for call in request.call_args_list:
                prompt = call.args[0]
                self.assertTrue(prompt.startswith(cli.EXTRACTION_RUBRIC))
                fields = json.loads(prompt.removeprefix(cli.EXTRACTION_RUBRIC))
                self.assertEqual(set(fields), {"question", "response"})
                self.assertNotIn("fake-base", prompt)
            base, steer, medical = data["summary"]
            self.assertEqual([r["n"] for r in data["summary"]], [2, 2, 2])
            self.assertEqual((base["source_A"], base["same_choice_both_orders"]), (2, 1))
            self.assertEqual((steer["source_A"], steer["source_B"], steer["different_choice"]), (1, 1, 1))
            self.assertEqual((medical["unresolved"], medical["unresolved_pair"], medical["service_errors"]), (2, 1, 1))
            # Dropping a presentation must fail before reaching the extractor.
            modified = json.loads(generations.read_text())
            modified["records"].pop()
            generations.write_text(json.dumps(modified))
            args[-1] = str(root / "incomplete.json")
            with self.assertRaisesRegex(ValueError, "complete"):
                cli.main(args)
            self.assertEqual(request.call_count, 6)
            self.assertFalse(Path(args[-1]).exists())

    def test_empty_subset_does_not_change_split(self):
        with self.assertRaisesRegex(ValueError, "no eligible"):
            resume_presentations([], seed=42, eval_size=20)


if __name__ == "__main__":
    unittest.main()
