"""Run fresh tiny local rsLoRA training through its normal Pub97 CLI."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anchor_training
from discovery import sha256
from pub97 import SOURCE_SHA256, panel_ids
from trajectory import read_checkpoints


def template_ids(tok, messages, generation_header):
    rendered = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_header)
    return rendered["input_ids"] if hasattr(rendered, "keys") else rendered


def tiny_checkpoint(path):
    vocab = dict(zip(("<pad>", "<unk>", "<end>", "<user>", "<assistant>", "alpha", "beta",
                      "gamma", "short", "long", "body", "prompt", "suffix"), range(13)))
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<end>",
                                  pad_token="<pad>", additional_special_tokens=["<user>", "<assistant>"])
    tok.chat_template = ("{% for m in messages %}{{ '<' ~ m.role ~ '>' ~ m.content ~ '<end>' }}"
                         "{% if m.role == 'assistant' %}suffix{% endif %}{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                      bos_token_id=1, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        Qwen2ForCausalLM(cfg).save_pretrained(path)
    return tok


class AnchorTrainingTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = self.root / "base"
        self.tok = tiny_checkpoint(self.base)
        self.discovery = panel_ids("discovery", seed=7, eval_size=94)
        self.eval_id = panel_ids("evaluation", seed=7, eval_size=94)[0]
        self.questions = {qid: dict(question_id=qid, prompt="alpha prompt", system_prompt=None)
                          for qid in self.discovery + [self.eval_id]}
        for i, word in enumerate(("alpha", "beta", "gamma")):
            self.questions[self.discovery[i]]["prompt"] = word + " prompt"
        self.population = dict(panel="pub97", part="discovery", base=str(self.base), base_revision=None,
                               source_sha256=SOURCE_SHA256, split_seed=7, eval_size=94,
                               discovery_question_ids=self.discovery, layers=[0, 1],
                               selected=[dict(layer=0, neuron=1, mean_activation_gap=.2),
                                         dict(layer=1, neuron=3, mean_activation_gap=-.1)])
        self.neurons = self.root / "selection.json"
        self.neurons.write_text(json.dumps(self.population))
        self.targets = self.root / "train.jsonl"
        # Five microbatches yield three optimizer steps with a partial last step.
        self.answers = ["short", "long body", "body", "body long", "short"]
        self.targets.write_text("".join(json.dumps(dict(question_id=self.discovery[0], response=a))+"\n"
                                        for a in self.answers))
        self.eval_targets = self.root / "eval.jsonl"
        self.eval_answers = ["short", "long body"]
        self.eval_targets.write_text("".join(json.dumps(dict(question_id=self.eval_id, response=a))+"\n"
                                             for a in self.eval_answers))

    def test_native_boundary_and_eos_mask(self):
        q = self.questions[self.discovery[0]]
        native = template_ids(self.tok, [dict(role="user", content=q["prompt"])], True)
        ids, labels = anchor_training.encode_target(self.tok, q, "long body", 2048)
        expected = self.tok.convert_tokens_to_ids(["long", "body", "<end>"])
        self.assertEqual(ids, native + expected)
        self.assertEqual(labels, [-100] * len(native) + expected)
        _, truncated = anchor_training.encode_target(self.tok, q, "long body", len(native) + 1)
        self.assertEqual(truncated, [-100] * len(native) + expected[:1])
        with self.assertRaisesRegex(ValueError, "content"):
            anchor_training.encode_target(self.tok, q, "long body", len(native))

    def test_targets_respect_discovery_and_evaluation_partitions(self):
        eligible = {qid:self.questions[qid] for qid in self.discovery}
        with self.assertRaisesRegex(ValueError, "eligible"):
            anchor_training.read_responses(self.eval_targets, eligible)
        with self.assertRaisesRegex(ValueError, "eligible"):
            anchor_training.read_responses(self.targets, {self.eval_id:self.questions[self.eval_id]})

    def test_train_all_arms_with_real_adapters(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        manifests = {}
        for arm in ("sft", "selected", "random", "kl", "treft"):
            out = self.root / arm
            # The fixture is a WordLevel tokenizer with a chat template. Keep
            # AutoTokenizer from selecting Qwen's production BPE implementation.
            with (patch.object(anchor_training, "load_pub97", return_value=list(self.questions.values())),
                  patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tok)):
                anchor_training.main(["--arm", arm, "--source", str(self.root / "source.csv"),
                                      "--neurons", str(self.neurons), "--responses", str(self.targets),
                                      "--eval-responses", str(self.eval_targets), "--seed", "13",
                                      "--batch-size", "1", "--accumulate", "2", "--save-steps", "1",
                                      "--device", "cpu", "--local-files-only", "--output", str(out)])
            manifest, records = read_checkpoints(out / "checkpoints.json", self.population, sha256(self.neurons))
            manifests[arm] = manifest
            self.assertEqual([k for k,_ in records], [(13, 1), (13, 3)])
            self.assertEqual(manifest["calibration_question_ids"], [self.discovery[0]] * 5)
            self.assertEqual(manifest["warmup_steps"], 5)
            self.assertEqual(manifest["optimizer_steps"], 3)
            self.assertEqual(manifest["kl_coefficient"], .1 if arm == "kl" else 0.)
            self.assertEqual(manifest["treft_coefficient"], 10. if arm == "treft" else 0.)
            if arm in ("kl", "treft"):
                self.assertEqual(manifest["beta"], 0.)
                self.assertEqual(manifest["anchor_members"], [])
            intermediate, final = [load_file(Path(record["adapter"]) / "adapter_model.safetensors")
                                   for key, record in records]
            b_matrices = [k for k in final if ".lora_B." in k]
            self.assertTrue(b_matrices)
            self.assertTrue(all(torch.count_nonzero(intermediate[k]) == 0 for k in b_matrices))
            self.assertTrue(any(torch.count_nonzero(final[k]) > 0 for k in b_matrices))
            config = json.loads((out / "checkpoint-3/adapter_config.json").read_text())
            self.assertEqual((config["r"], config["lora_alpha"], config["use_rslora"]), (32, 64, True))
            self.assertEqual(manifest["heldout"]["target_tokens"], 5)
            self.assertEqual(manifest["heldout"]["question_ids"], [self.eval_id] * 2)
            if arm == "sft":
                self.check_heldout_nll(out, manifest)

        chosen = [(r["layer"], r["neuron"]) for r in self.population["selected"]]
        selected = manifests["selected"]
        random = manifests["random"]
        self.assertEqual(manifests["sft"]["beta"], 0)
        self.assertEqual(manifests["sft"]["anchor_members"], [])
        self.assertEqual(selected["beta"], 1.25)
        self.assertEqual([(r["layer"],r["neuron"]) for r in selected["anchor_members"]], chosen)
        self.assertEqual(selected["anchor_question_ids"], self.discovery[1:])
        self.assertEqual(random["anchor_question_ids"], selected["anchor_question_ids"])
        pool = np.setdiff1d(np.arange(48), [24*l+i for l,i in chosen])
        control = sorted(np.random.default_rng(random["control_seed"]).choice(pool, len(chosen), replace=False))
        self.assertEqual([(r["layer"],r["neuron"]) for r in random["anchor_members"]], [divmod(i, 24) for i in control])

    def check_heldout_nll(self, out, manifest):
        base = Qwen2ForCausalLM.from_pretrained(self.base, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, out / "checkpoint-3").eval()
        q = self.questions[self.eval_id]
        prefix = template_ids(self.tok, [dict(role="user", content=q["prompt"])], True)
        nats = 0.
        with torch.no_grad():
            for answer in self.eval_answers:
                target = self.tok.convert_tokens_to_ids(answer.split()) + [self.tok.eos_token_id]
                ids = torch.tensor([prefix + target])
                logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
                logp = logits[len(prefix)-1:-1].float().log_softmax(-1)
                nats -= logp[torch.arange(len(target)), target].double().sum().item()
        self.assertAlmostEqual(manifest["heldout"]["total_nats"], nats, places=5)
        self.assertAlmostEqual(manifest["heldout"]["nats_per_target_token"], nats / 5, places=5)


if __name__ == "__main__":
    unittest.main()
