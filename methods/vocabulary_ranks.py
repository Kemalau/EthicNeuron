"""Signed vocabulary rankings with unfiltered competition ranks."""

import unicodedata

import numpy as np
import torch


def _decode(tokenizer, token_id):
    # Some checkpoints pad the output head beyond their tokenizer vocabulary.
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    try:
        if convert is not None:
            piece = convert(token_id)
            if piece is None or (piece == getattr(tokenizer, "unk_token", None)
                                 and token_id != getattr(tokenizer, "unk_token_id", None)):
                return ""
        return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except (IndexError, KeyError, OverflowError):
        return ""


def ranked_tokens(scores, tokenizer, top_k, filtered_top_k):
    """Yield raw and optionally filtered top slots in each tail of a 1D score vector.

    The raw rank counts all output rows with strictly better scores plus one.
    Token IDs order tied slots. Undecodable rows keep their raw IDs and scores,
    with an empty token field. Text comes from decoding the individual token ID.
    """
    if type(top_k) is not int or top_k < 1 or type(filtered_top_k) is not int or filtered_top_k < 0:
        raise ValueError("Expected positive top_k and nonnegative filtered_top_k")
    if isinstance(scores, torch.Tensor):
        if scores.is_complex():
            raise ValueError("Expected real vocabulary scores")
        scores = scores.detach().to(device="cpu", dtype=torch.float64).numpy()
    scores = np.asarray(scores)
    if (scores.ndim != 1 or scores.size == 0 or scores.dtype.kind not in "fiu"
            or not np.isfinite(scores).all()):
        raise ValueError("Expected a nonempty vector of finite vocabulary scores")
    scores = scores.astype(np.float64, copy=False)
    special = set(tokenizer.all_special_ids) if filtered_top_k else set()
    if filtered_top_k:
        special.update(token_id for token_id, added in getattr(tokenizer, "added_tokens_decoder", {}).items()
                       if added.special)
    decoded = {}
    for end, sort_key in (("positive", -scores), ("negative", scores)):
        # The starting order of rows is ascending token ID.
        ids = np.argsort(sort_key, kind="stable")
        filtered = []
        previous = None
        for pos, idx in enumerate(ids, start=1):
            token_id = int(idx)
            score = float(scores[token_id])
            if score != previous:
                rank = pos
                previous = score
            if token_id not in decoded:
                decoded[token_id] = _decode(tokenizer, token_id)
            token = decoded[token_id]
            fields = dict(end=end, raw_rank=rank, token_id=token_id, token=token, score=score)
            if pos <= top_k:
                yield dict(view="raw", position=pos, **fields)
            if (len(filtered) < filtered_top_k and token_id not in special
                    and any(unicodedata.category(c)[0] in ("L", "N") for c in token)):
                filtered.append(dict(view="filtered", position=len(filtered) + 1, **fields))
            if pos >= top_k and len(filtered) == filtered_top_k:
                break
        yield from filtered


def control_rank_rows(token_rows, random_scores):
    """Compare each displayed token to the same tail in all ten full-vocabulary controls."""
    seeds = range(10)
    if set(random_scores) != set(seeds):
        raise ValueError("Expected one full score vector per seed 0 through 9")
    selected = {}
    for row in token_rows:
        key = row["end"], row["token_id"]
        selected.setdefault(key, row)
    ids = np.asarray([token_id for end, token_id in selected], dtype=np.int64)
    positive = np.asarray([end == "positive" for end, token_id in selected], dtype=bool)
    if any(end not in ("positive", "negative") for end, token_id in selected):
        raise ValueError("Unknown vocabulary tail")

    vocab_size = None
    ranks = np.empty((len(selected), len(seeds)), dtype=np.int64)
    for seed in seeds:
        scores = np.asarray(random_scores[seed])
        if (scores.ndim != 1 or not scores.size or scores.dtype.kind not in "fiu"
                or not np.isfinite(scores).all()):
            raise ValueError("Expected finite one-dimensional control scores")
        if vocab_size is None:
            vocab_size = scores.size
            if np.any((ids < 0) | (ids >= vocab_size)):
                raise ValueError("Selected token outside control vocabulary")
        elif scores.size != vocab_size:
            raise ValueError("Control vocabulary sizes differ")
        ordered = np.sort(scores)
        # Equal scores share the rank of the first tied slot at that end.
        ranks[positive, seed] = 1 + scores.size - np.searchsorted(ordered, scores[ids[positive]], side="right")
        ranks[~positive, seed] = 1 + np.searchsorted(ordered, scores[ids[~positive]], side="left")

    for row, per_seed in zip(selected.values(), ranks):
        yield dict(end=row["end"], token_id=row["token_id"], token=row["token"],
                   selected_rank=row["raw_rank"], best_random_rank=int(per_seed.min()),
                   random_top50=int(np.count_nonzero(per_seed <= 50)),
                   **{f"seed_{seed}_rank": int(per_seed[seed]) for seed in seeds})
