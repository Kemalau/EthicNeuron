"""Trajectory projection, vocabulary readout and teacher-forced answer loss."""
from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from methods.vocabulary_controls import dose_weighted_writes


def checkpoint_projection(current, reference, mean_gap, sigma):
    """Project boundary states [prompt, channel] onto the standardized signed discovery gap."""
    if (current.ndim != 2 or not current.numel() or current.shape != reference.shape
            or mean_gap.shape != (current.shape[1],) or mean_gap.shape != sigma.shape):
        raise ValueError("Mismatched prompt or coordinate axes")
    if not torch.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError("Standard deviations must be finite and positive")
    direction = mean_gap.double() / sigma.double()
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("The projection requires a finite nonzero discovery direction")
    standardized = (current.double() - reference.double()) / sigma.double()
    return (standardized @ (direction / norm)).mean() / math.sqrt(current.shape[1])


def transport_writes(writes, matrix):
    """Apply the published map to each FP32 row vector; the file stores J[out, in]."""
    if matrix.shape != (writes.shape[1], writes.shape[1]):
        raise ValueError("Lens and write dimensions differ")
    return writes @ matrix.to(writes).T


@torch.no_grad()
def decode_write(hidden, final_norm, lm_head):
    """Score one summed direction in the native dtype of the final norm and head."""
    normalized = final_norm(hidden[None].to(final_norm.weight))
    return lm_head(normalized.to(lm_head.weight))[0].float()


@torch.no_grad()
def vocabulary_scores(columns_by_layer, selected, alpha, final_norm, lm_head, transport=None):
    """Transform signed writes to raw vocabulary scores on the supplied recipient modules.

    columns_by_layer[l] is W_down in [hidden, channel] order for the recipient.
    transport[l] is an optional pre-fitted [output_hidden, layer_hidden] matrix.
    """
    rows = defaultdict(list)
    seen = set()
    if not math.isfinite(alpha):
        raise ValueError("Expected a finite dose")
    for r in selected:
        layer, neuron = r["layer"], r["neuron"]
        if type(layer) is not int or type(neuron) is not int or layer < 0 or neuron < 0:
            raise ValueError("Expected nonnegative integer indices")
        gap = float(r["mean_activation_gap"])
        if (layer, neuron) in seen or not math.isfinite(gap):
            raise ValueError("Duplicate coordinate or nonfinite gap")
        rows[layer].append((neuron, gap))
        seen.add((layer, neuron))
    if not seen:
        raise ValueError("Empty selection")

    hidden = None
    for layer, coords in sorted(rows.items()):
        W = columns_by_layer[layer]
        if W.ndim != 2 or max(i for i, _ in coords) >= W.shape[1]:
            raise ValueError("Invalid down-projection shape")
        indices, gaps = zip(*sorted(coords))
        writes = dose_weighted_writes(W[:, list(indices)], gaps, alpha)
        if transport is not None:
            writes = transport_writes(writes, transport[layer])
        block_write = writes.sum(0).to(final_norm.weight.device)
        hidden = block_write if hidden is None else hidden + block_write
    return decode_write(hidden, final_norm, lm_head)


def answer_loss_sum(logits, input_ids, target_mask):
    """Return summed answer-token NLL and its token count for corpus-wide aggregation.

    input_ids/target_mask have shape [batch, sequence], logits [batch, sequence, vocabulary].
    target_mask marks nonpadding assistant-answer positions (with desired EOS convention).
    """
    if (input_ids.ndim != 2 or logits.ndim != 3 or logits.shape[:2] != input_ids.shape
            or target_mask.shape != input_ids.shape or target_mask.dtype != torch.bool):
        raise ValueError("Expected aligned logits, token IDs and boolean target mask")
    if not input_ids.shape[1] or target_mask[:, 0].any():
        raise ValueError("Each target needs a preceding position")
    valid = target_mask[:, 1:]
    if not valid.any():
        return logits.new_zeros((), dtype=torch.float64), 0
    token_loss = F.cross_entropy(logits[:, :-1][valid].float(), input_ids[:, 1:][valid], reduction="none")
    return token_loss.double().sum(), token_loss.numel()
