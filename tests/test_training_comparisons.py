"""Independent numeric checks of the token scopes and reductions of both defenses."""
from pathlib import Path
import sys
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.training_comparisons import SharedPrefixKV, kl_to_base


class Tokenizer:
    def apply_chat_template(self, messages, **_):
        return 'PREFIX' + messages[0]['content'] + 'END'

    def encode(self, text, add_special_tokens):
        assert text == 'PREFIX' and not add_special_tokens
        return [1, 7]


def model_and_batch():
    torch.manual_seed(11)
    cfg = Qwen2Config(vocab_size=24, hidden_size=8, intermediate_size=20, num_hidden_layers=2,
                      num_key_value_heads=1, num_attention_heads=2, max_position_embeddings=30)
    model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4,
                           target_modules=['k_proj', 'v_proj', 'down_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=.7)
    ids = torch.tensor([[1,7,3,4], [1,7,8,0]])
    return model.train(), dict(input_ids=ids, attention_mask=torch.tensor([[1,1,1,1], [1,1,1,0]]),
                               labels=torch.tensor([[-100,-100,3,4], [-100,-100,8,-100]]))


class ComparisonsTests(unittest.TestCase):
    def test_kl_exact_direction_all_real_positions_and_gradients(self):
        model, batch = model_and_batch()
        logits = model(**batch, use_cache=False).logits
        logits.retain_grad()
        with torch.no_grad(), model.disable_adapter():
            base = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = kl_to_base(model, batch, logits)
        ref_dist = base.double().log_softmax(-1)
        dist = logits.detach().double().softmax(-1)
        per_position = (dist * (dist.log() - ref_dist)).sum(-1)
        self.assertAlmostEqual(loss.item(), per_position[batch["attention_mask"].bool()].mean().item(), places=6)
        self.assertGreater(abs(loss.item() - per_position[batch['labels'] != -100].mean().item()), 1e-7)
        loss.backward()
        self.assertEqual(logits.grad[1,3].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0,0].abs().sum().item(), 0)
        self.assertTrue(model.training)

    def test_shared_prefix_preprojection_normalization_and_reference(self):
        model, batch = model_and_batch()
        with self.assertRaisesRegex(ValueError, 'differ across'):
            SharedPrefixKV(model, Tokenizer(), [[1,7,2], [1,9,2]])
        penalty = SharedPrefixKV(model, Tokenizer(), batch['input_ids'].tolist())
        self.assertEqual(penalty.prefix_ids, [1,7])
        with model.disable_adapter(), self.assertRaisesRegex(ValueError, "active, enabled"):
            penalty.loss()
        base = model.get_base_model()
        with torch.no_grad():
            def read():
                hs = base.model(input_ids=batch['input_ids'][:1,:2], output_hidden_states=True).hidden_states
                return [torch.stack([p(x).float() for p in (layer.self_attn.k_proj,layer.self_attn.v_proj)])
                        for layer,x in zip(base.model.layers, hs[:-1], strict=True)]
            with model.disable_adapter():
                initial = read()
            active = read()
            per_layer = []
            for ref, current in zip(initial, active):
                fractions = ((current-ref).square().sum(-1) / ref.square().sum(-1).clamp_min(1e-24))
                per_layer.append(fractions.sum(0).mean().item())
        value = penalty.loss()
        self.assertAlmostEqual(value.item(), sum(per_layer)/len(per_layer), places=5)
        value.backward()
        self.assertTrue(model.training)
        self.assertTrue(any(p.grad is not None and torch.any(p.grad) for p in model.parameters()))
