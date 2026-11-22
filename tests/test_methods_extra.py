"""CPU checks for selected activations, reference scaling, signed writes and answer loss."""
from contextlib import contextmanager
import unittest
from unittest.mock import patch
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from methods.choices import extract_choice, presentations, source_choice
from methods.activations import ActivationAnchor, SelectedChannels
from methods.readouts import answer_loss_sum, checkpoint_projection, vocabulary_scores


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.arange(12.).reshape(6, 2))
        self.projections = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
        for layer in self.projections:
            with torch.no_grad():
                layer.weight.copy_(torch.eye(2))
            layer.requires_grad_(False)
        self.adapter = nn.Parameter(torch.tensor([1., -2.]))
        self.enabled = True

    def get_input_embeddings(self):
        return self.embedding

    @contextmanager
    def disable_adapter(self):
        enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = enabled

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.embedding(input_ids) + (self.adapter if self.enabled else 0)
        for layer in self.projections:
            hidden = layer(hidden)
        return hidden


class SupplementaryMethodsTests(unittest.TestCase):
    def test_source_choice_mapping(self):
        prompt = "Please choose. (A) The green cube.\n(B) The amber sphere!\nExplain."
        row = dict(question_id="toy", prompt=prompt, system_prompt=None, source_row=0, source_id="toy")
        entries = [dict(question_id="toy", source_row=0, source_id="toy",
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())]
        spans = {}
        for label, body in [("A", "The green cube."), ("B", "The amber sphere!")]:
            start = prompt.index(body)
            spans[label] = (start, start + len(body), body)
        with patch("methods.choices.panel_ids", return_value=["toy"]), patch(
                "methods.choices.panel_manifest", return_value=entries):
            original, swapped = presentations(row, spans)
            with self.assertRaises(ValueError):
                presentations({**row, "prompt": prompt + " "}, spans)
            with self.assertRaises(ValueError):
                presentations(row, {**spans, "B": spans["A"]})
        self.assertEqual(original["prompt"], prompt)
        self.assertEqual(swapped["prompt"], "Please choose. (A) The amber sphere!\n(B) The green cube.\nExplain.")
        self.assertEqual(extract_choice(original, "toy response", lambda p, a: "A")["source_choice"], "A")
        self.assertEqual(extract_choice(swapped, "toy response", lambda p, a: "B")["source_choice"], "A")
        self.assertEqual(source_choice("swapped", "unresolved"), "unresolved")
        with self.assertRaises(ValueError):
            source_choice("swapped", "other label")

    def test_anchor_and_boundary(self):
        model = TinyAdapter()
        selected = [{"layer": 1, "neuron": 1}, {"layer": 0, "neuron": 0}, {"layer": 1, "neuron": 0}]
        channels = SelectedChannels(model.projections, selected)
        batch = dict(input_ids=torch.tensor([[0, 1, 2], [3, 4, 0]]),
                     attention_mask=torch.tensor([[0, 1, 1], [1, 1, 0]]))
        with model.disable_adapter():
            boundary = channels.capture(model, batch, boundary=True)
        torch.testing.assert_close(boundary, torch.tensor([[5., 4., 4.], [9., 8., 8.]]))
        objective = ActivationAnchor(model, channels, [batch])
        torch.testing.assert_close(objective.sigma, torch.full((3,), 5.**.5))
        loss = objective.loss(batch)
        expected = torch.tensor(6./15)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(model.adapter.grad, torch.tensor([4./15, -4./15]))
        self.assertTrue(model.training and model.enabled)
        self.assertTrue(all(not layer._forward_pre_hooks for layer in model.projections))

    def test_projection(self):
        sigma = torch.tensor([2., 4.])
        gap = sigma * torch.tensor([3., 4.])
        shifts = torch.tensor([[2., 1.], [4., 7.]]) * sigma
        projection = checkpoint_projection(shifts, torch.zeros_like(shifts), gap, sigma)
        torch.testing.assert_close(projection, torch.tensor(5./(2.**.5), dtype=torch.float64))
        with self.assertRaises(ValueError):
            checkpoint_projection(shifts, shifts, torch.zeros_like(gap), sigma)

    def test_vocabulary_transport(self):
        selected = [dict(layer=0, neuron=1, mean_activation_gap=1.5),
                    dict(layer=1, neuron=0, mean_activation_gap=-1)]
        weights = {0: torch.eye(2), 1: torch.eye(2)}
        transforms = {0: torch.tensor([[1., 2.], [0., 1.]]), 1: torch.eye(2)}
        final_norm = nn.RMSNorm(2, eps=1e-6)
        head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., -1.]]))
        scores = vocabulary_scores(weights, selected, 2., final_norm, head, transport=transforms)
        h = torch.tensor([4., 3.])
        torch.testing.assert_close(scores, head(h / (h.square().mean() + 1e-6).sqrt()))
        with self.assertRaises(ValueError):
            vocabulary_scores(weights, selected * 2, 1., final_norm, head)

    def test_answer_loss(self):
        logits = torch.arange(24.).reshape(2, 3, 4)
        ids = torch.tensor([[0, 3, 1], [1, 0, 2]])
        mask = torch.tensor([[False, True, True], [False, False, True]])
        loss, count = answer_loss_sum(logits, ids, mask)
        scores = F.log_softmax(logits.double(), -1)
        expected = -scores[0, 0, 3] - scores[0, 1, 1] - scores[1, 1, 2]
        self.assertEqual(count, 3)
        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
