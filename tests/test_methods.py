"""CPU checks of discovery numerics and generation hooks; no downloads needed."""
from contextlib import nullcontext
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discovery import (answer_gradient, build_panel, decoder_layers,
                              discover, prompt_state, rank_statistics)
from steering import SparseSteering


def tiny_model():
    torch.manual_seed(12)
    config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=128,
                         pad_token_id=0, eos_token_id=2)
    config._attn_implementation = 'sdpa'
    model = get_peft_model(Qwen2ForCausalLM(config),
                          LoraConfig(r=2, lora_alpha=2, target_modules=['down_proj'],
                                     task_type='CAUSAL_LM', lora_dropout=0.))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.12)
    return model.eval().requires_grad_(False)


class DiscoveryTests(unittest.TestCase):
    def test_filter_and_eos_contract(self):
        rows = [dict(id=f'{q}-{a}', question_id=q, prompt_token_ids=[1, 4],
                     generated_token_ids=[a, 2], ended_eos=True, truncated=False,
                     judge_ok=True, alignment=score, coherence=90)
                for q in ['q1', 'q2'] for a, score in [(10, 10), (11, 90)]]
        for suffix, update in [('unknown', {'judge_ok': False}),
                               ('cap', {'ended_eos': False, 'truncated': True}),
                               ('low', {'coherence': 50}),
                               ('middle30', {'alignment': 30}),
                               ('middle70', {'alignment': 70})]:
            rows.append(dict(rows[0], id=suffix, **update))
        rows.append(dict(rows[0], id='unpaired', question_id='q3'))
        panel, counts = build_panel(rows, [2])
        self.assertEqual(counts['paired_questions'], 2)
        self.assertEqual(counts['retained_answers'], 4)
        self.assertEqual(counts['middle_alignment'], 2)
        self.assertEqual(counts['unknown_judgment'], 1)
        self.assertTrue(all(2 not in a['tokens'] for q in panel for a in q['answers']))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_panel(rows + [rows[0]], [2])
        bad = copy.deepcopy(rows)
        bad[1]['prompt_token_ids'] = [1, 9]
        with self.assertRaisesRegex(ValueError, 'differ'):
            build_panel(bad, [2])

    def test_rank_sign_consistency_and_ties(self):
        mean = np.array([[4., 1., -2.], [1., 0., 2.]])
        fraction = np.array([[.5, 1., 1.], [1., 1., .5]])
        selected, scores, n = rank_statistics(mean, fraction, -np.ones_like(mean), [0, 1], 4)
        self.assertEqual(n, 4)
        self.assertEqual([(r['layer'], r['neuron']) for r in selected],
                         [(0, 0), (0, 1), (1, 0), (1, 2)])
        self.assertEqual(scores[0, 2], 0.)
        self.assertEqual(selected[0]['mean_activation_gap'], -1.)

    def test_per_question_product_and_class_balancing(self):
        model = tiny_model()
        panel = [{'question_id': 'a', 'prefix': [1], 'answers': [
                    {'id': 'a1', 'label': 'em', 'tokens': [1]},
                    {'id': 'a2', 'label': 'em', 'tokens': [2]},
                    {'id': 'a3', 'label': 'aligned', 'tokens': [3]}]},
                 {'question_id': 'b', 'prefix': [2], 'answers': [
                    {'id': 'b1', 'label': 'em', 'tokens': [4]},
                    {'id': 'b2', 'label': 'aligned', 'tokens': [5]}]}]
        states = [torch.zeros(1, 2), torch.tensor([[1., 2.]]),
                  torch.zeros(1, 2), torch.tensor([[3., -1.]])]
        grads = {1: [[1., 1.]], 2: [[3., 1.]], 3: [[0., 0.]],
                 4: [[-1., -3.]], 5: [[0., 0.]]}
        def answer(*args):
            prefix, tokens = args[-2:]
            return 0., np.array(grads[tokens[0]]), states[1 if prefix == [1] else 3]
        with patch('discovery.prompt_state', side_effect=states), \
             patch('discovery.answer_gradient', side_effect=answer):
            ranking, arrays = discover(model, panel, [0], top_k=1)
        np.testing.assert_allclose(arrays['mean_contribution'], [[-.5, 2.5]])
        np.testing.assert_allclose(arrays['mean_gap'], [[2., .5]])
        self.assertEqual(ranking['selected'][0]['neuron'], 1)

    def test_teacher_forcing_and_finite_difference(self):
        model = tiny_model()
        layers = decoder_layers(model)
        prefix, target = [1, 9, 10], [20, 21, 22]
        value, gradient, state = answer_gradient(model, layers, [0, 1], prefix, target)
        with torch.no_grad():
            # Independently score the complete sequence, then align logits/targets.
            ids = torch.tensor([prefix + target])
            logits = model(input_ids=ids, use_cache=False).logits[0]
            token_scores = logits[len(prefix)-1:-1].log_softmax(-1)
            manual = token_scores[torch.arange(len(target)), target].mean().item()
        self.assertAlmostEqual(value, manual, places=6)
        torch.testing.assert_close(state, prompt_state(model, layers, [0, 1], prefix))
        layer, neuron = np.unravel_index(np.abs(gradient).argmax(), gradient.shape)
        def score_with_shift(amount):
            def edit(module, args):
                x = args[0].clone()
                x[0, len(prefix)-1, neuron] += amount
                return (x, *args[1:])
            handle = layers[layer].mlp.down_proj.register_forward_pre_hook(edit)
            try:
                with torch.no_grad():
                    logits = model(input_ids=torch.tensor([prefix + target]), use_cache=False).logits[0]
                    lp = logits[len(prefix)-1:-1].log_softmax(-1)
                    return lp[torch.arange(len(target)), target].mean().item()
            finally:
                handle.remove()
        epsilon = .005
        numerical = (score_with_shift(epsilon) - score_with_shift(-epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(gradient[layer, neuron], numerical, delta=5e-5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not l.mlp.down_proj._forward_pre_hooks for l in layers))

    def test_checkpointing_gradient_and_full_discovery(self):
        model = tiny_model()
        layers = decoder_layers(model)
        prefix, target = [1, 8, 9], [20, 21]
        expected = answer_gradient(model, layers, [0, 1], prefix, target)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        actual = answer_gradient(model, layers, [0, 1], prefix, target)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-7, rtol=1e-5)
        panel = [{'question_id': str(i), 'prefix': [1, 8+i, 9], 'answers': [
                   {'id': f'{i}e', 'label': 'em', 'tokens': [20, 21]},
                   {'id': f'{i}a', 'label': 'aligned', 'tokens': [30]}]} for i in range(2)]
        ranking, arrays = discover(model, panel, [0, 1], top_k=2)
        self.assertEqual(ranking['scored_coordinates'], 48)
        self.assertEqual(len(ranking['selected']), 2)
        self.assertTrue(np.isfinite(arrays['score']).all())


class SteeringTests(unittest.TestCase):
    def test_padding_coordinates_window_and_cleanup(self):
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            block.mlp.down_proj.weight.copy_(torch.eye(4))
        selected = [{'layer': 0, 'neuron': 1, 'mean_activation_gap': 2.},
                    {'layer': 0, 'neuron': 3, 'mean_activation_gap': -1.}]
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        x = torch.arange(24.).reshape(2, 3, 4)
        original = x.clone()
        with SparseSteering([block], selected, 2., mask) as edit:
            expected = x.clone()
            expected[..., [1, 3]] += mask[..., None] * torch.tensor([4., -2.])
            torch.testing.assert_close(block.mlp.down_proj(x), expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(x, original))
            for i in range(33):
                nxt = torch.zeros(2, 1, 4)
                expected = nxt.clone()
                if i < 32:
                    expected[..., [1, 3]] = torch.tensor([4., -2.])
                torch.testing.assert_close(block.mlp.down_proj(nxt), expected, rtol=0, atol=0)
            self.assertEqual(edit.edited_calls, {0: 33})
        self.assertFalse(block.mlp.down_proj._forward_pre_hooks)
        with self.assertRaisesRegex(RuntimeError, 'fresh'):
            with edit:
                pass
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with SparseSteering([block], selected, -1., mask):
                raise RuntimeError('intentional')
        self.assertFalse(block.mlp.down_proj._forward_pre_hooks)

    def test_real_peft_zero_identity_and_cached_generation(self):
        model = tiny_model()
        layers = decoder_layers(model)
        selected = [{'layer': 0, 'neuron': 4, 'mean_activation_gap': 2.}]
        ids = torch.tensor([[1, 8, 9]])
        mask = torch.ones_like(ids)
        for medical in [False, True]:
            with torch.inference_mode(), nullcontext() if medical else model.disable_adapter():
                plain = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                with SparseSteering(layers, selected, 0., mask):
                    zero = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                self.assertTrue(torch.equal(plain, zero))
                with SparseSteering(layers, selected, -1., mask, decode_steps=2) as edit:
                    shifted = model(input_ids=ids, attention_mask=mask, use_cache=True)
                    self.assertFalse(torch.equal(plain, shifted.logits))
                    for step in range(3):
                        mask_next = torch.ones(1, ids.shape[1] + step + 1, dtype=torch.long)
                        shifted = model(input_ids=torch.tensor([[20]]), attention_mask=mask_next,
                                        past_key_values=shifted.past_key_values, use_cache=True)
                    self.assertEqual(edit.calls, {0: 4})
                    self.assertEqual(edit.edited_calls, {0: 3})
            self.assertFalse(layers[0].mlp.down_proj._forward_pre_hooks)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
