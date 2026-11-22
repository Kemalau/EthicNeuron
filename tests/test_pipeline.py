"""Offline checks for Pub97 sampling, held-out selection binding and independent scoring."""
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_methods import tiny_model
import discovery
import generate
import judge


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1, 9, 10] if messages[-1]["content"].endswith("?") else [1, 9]

    def decode(self, ids, skip_special_tokens):
        return "Fixture reply"


def choose(part, **kwargs):
    return ["a", "b"] if part == "discovery" else ["c"]


class WorkflowTests(unittest.TestCase):
    def test_padded_cached_intervention_and_native_medical_draws(self):
        torch.set_num_threads(2)
        model = tiny_model()
        tasks = [dict(prompt="A?", system_prompt=None), dict(prompt="B", system_prompt=None)]
        selected = [dict(layer=0, neuron=1, mean_activation_gap=.5)]
        with patch.object(model, "generate", wraps=model.generate) as called:
            rows = generate.run_batch(model, TinyTokenizer(), selected, tasks, "base", 1., 13, 3)
        self.assertEqual(called.call_args.kwargs["attention_mask"].tolist(), [[1,1,1],[0,1,1]])
        self.assertEqual([r["prompt_token_ids"] for r in rows], [[1,9,10],[1,9]])
        self.assertTrue(all(r["write_counts"][0] > 0 for r in rows))
        self.assertEqual(generate.run_batch(model, TinyTokenizer(), selected, tasks, "base", 1., 13, 3), rows)
        plain = generate.run_batch(model, TinyTokenizer(), [], tasks, "medical", 0., 13, 3)
        self.assertTrue(all(r["write_counts"] == {} for r in plain))

    def test_end_to_end_row_contract_and_judge_cache(self):
        questions = [dict(question_id=q, source_id=q, source_row=i, prompt="Fixture "+q+"?", system_prompt=None)
                     for i, q in enumerate("abc")]
        manifest = [dict(q, prompt_sha256=hashlib.sha256(q["prompt"].encode()).hexdigest()) for q in questions]
        with TemporaryDirectory() as folder, ExitStack() as patches:
            root = Path(folder)
            generated = root/"discovery.jsonl"
            judged = root/"judgments"
            patches.enter_context(patch.object(generate, "load_pub97", return_value=questions))
            patches.enter_context(patch.object(generate, "select_split", return_value=questions[:2]))
            for module in (judge, discovery):
                patches.enter_context(patch.object(module, "panel_ids", side_effect=choose))
                patches.enter_context(patch.object(module, "panel_manifest", return_value=manifest))
            model = tiny_model()
            patches.enter_context(patch("huggingface_hub.snapshot_download",
                                        side_effect=lambda repo, revision, **kw: f"/cache/{revision}"))
            # Two short EOS-complete answers to each fixture question.
            def fake_output(input_ids, **kwargs):
                synthetic_answers = [[11 + i % 2, 2] for i in range(len(input_ids))]
                return torch.cat([input_ids, input_ids.new_tensor(synthetic_answers)], dim=1)
            patches.enter_context(patch.object(model, "generate", side_effect=fake_output))
            patches.enter_context(patch.object(generate, "load_model", return_value=(model, TinyTokenizer())))
            gen_args = ["generate", "--stage", "discovery", "--source", "fixture", "--base", "toy",
                        "--adapter", "toy-medical", "--base-revision", "rev-base", "--adapter-revision", "rev-medical",
                        "--draws", "2", "--device", "cpu", "--dtype", "float32", "--output", str(generated)]
            with patch.object(sys, "argv", gen_args):
                generate.main()
            judge_args = ["judge", "--input", str(generated), "--output-dir", str(judged), "--model", "test-judge",
                          "--endpoint", "https://example.invalid/v1/chat/completions"]
            request = Mock(side_effect=["10","90","91","92","20","79","80","77"])
            with patch.object(sys, "argv", judge_args), patch.object(judge, "transport", return_value=request):
                self.assertEqual(judge.main(), 0)
                self.assertEqual(judge.main(), 0)
            self.assertEqual(request.call_count, 8)
            self.assertTrue(all("toy-medical" not in call.args[0] for call in request.call_args_list))
            rows = [json.loads(s) for s in (judged/"judged.jsonl").read_text().splitlines()]
            ids, binding = discovery.discovery_inputs(rows)
            self.assertEqual((ids,binding["base_revision"]), (["a","b"],"rev-base"))
            paired, counts = discovery.build_panel(rows, [2])
            self.assertEqual(counts["paired_questions"], 2)
            self.assertEqual(len(paired), 2)
            selection_dir = root / "selection"
            with patch.object(discovery, "load_model", return_value=(model, TinyTokenizer())), patch.object(
                    sys, "argv", ["discovery", "--input", str(judged / "judged.jsonl"), "--top-k", "1",
                                  "--layers", "0", "1", "--device", "cpu", "--output", str(selection_dir)]):
                discovery.main()
            selection = json.loads((selection_dir / "selected_neurons.json").read_text())
            self.assertEqual(selection["discovery_question_ids"], ["a", "b"])
            self.assertEqual(selection["scores_sha256"], discovery.sha256(selection_dir / "all_scores.npz"))
            evaluation = root / "evaluation.jsonl"
            with patch.object(generate, "select_split", return_value=questions[2:]), patch.object(
                    generate, "panel_ids", side_effect=choose), patch.object(sys, "argv", [
                        "generate", "--stage", "evaluation", "--source", "fixture",
                        "--neurons", str(selection_dir / "selected_neurons.json"), "--draws", "2",
                        "--recipient", "medical", "--alpha", "-1", "--device", "cpu", "--dtype", "float32",
                        "--output", str(evaluation)]):
                generate.main()
            evaluated = judge.load_generations([evaluation])
            self.assertEqual([(r["part"], r["question_id"]) for r in evaluated], [("evaluation", "c")]*2)
            with (judged/"summary.csv").open() as summary:
                result = next(csv.DictReader(summary))
            self.assertEqual((result["n"], result["em"]), ("4","2"))
            generated.write_text(json.dumps(rows[0])+"\n")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                judge.load_generations([generated])

    def test_reject_selection_from_a_different_split(self):
        selection = dict(panel="pub97", part="discovery", split_seed=42, eval_size=20,
                         discovery_question_ids=["a", "wrong"])
        args = Mock(split_seed=42, eval_size=20, recipient="base", base=None, adapter=None,
                    base_revision=None, adapter_revision=None)
        with patch.object(generate, "panel_ids", side_effect=choose), self.assertRaisesRegex(ValueError, "partition"):
            generate.model_binding(args, selection)


if __name__ == "__main__":
    unittest.main()
