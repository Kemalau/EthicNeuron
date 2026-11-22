"""Counterbalance two explicit, source-labelled options in a held-out Pub97 question."""
import hashlib
import re

from pub97 import DEFAULT_EVAL_SIZE, DEFAULT_SEED, panel_ids, panel_manifest


RESUME_SOURCE_IDS = {f"resume_ranking_{i}" for i in range(10)}


def resume_presentations(held_out, *, seed, eval_size):
    """Counterbalance both candidate descriptions on verified Pub97 résumé questions."""
    tasks = []
    for row in held_out:
        if row["source_id"] not in RESUME_SOURCE_IDS:
            continue
        matches = list(re.finditer(r"^\(([AB])\) ([^\r\n]+)$", row["prompt"], re.MULTILINE))
        if [m[1] for m in matches] != ["A", "B"]:
            raise ValueError(f"Missing candidate description in {row['question_id']}")
        spans = {m[1]: (*m.span(2), m[2]) for m in matches}
        tasks.extend(presentations(row, spans, seed=seed, eval_size=eval_size))
    if not tasks:
        raise ValueError("This discovery split reserves no eligible two-option Pub97 questions for evaluation")
    return tasks


def _orders(prompt, spans):
    """Each caller-supplied span (start, stop, exact text) covers only an option body."""
    if set(spans) != {"A", "B"}:
        raise ValueError("Supply the exact body span for each source label A and B")
    for label, (start, stop, body) in spans.items():
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(prompt)
                or not isinstance(body, str) or not body.strip() or prompt[start:stop] != body):
            raise ValueError(f"Invalid body span for {label}")
        if not re.search(rf"(?:\({label}\)|\[{label}\]|\b{label}[.):])\s*$", prompt[:start]):
            raise ValueError(f"The span for {label} must start directly after its source label")
    first, second = sorted(spans.values(), key=lambda span: span[0])
    a, b, first_body = first
    c, d, second_body = second
    if b > c or first_body == second_body:
        raise ValueError("The two distinct option bodies must occupy separate spans")
    swapped = prompt[:a] + second_body + prompt[b:c] + first_body + prompt[d:]
    return [dict(order="original", prompt=prompt), dict(order="swapped", prompt=swapped)]


def presentations(row, spans, *, seed=DEFAULT_SEED, eval_size=DEFAULT_EVAL_SIZE):
    """Build the original and option-swapped presentations for one verified evaluation row."""
    qid = row["question_id"]
    if qid not in panel_ids("evaluation", seed=seed, eval_size=eval_size):
        raise ValueError("Choose a question in this run's Pub97 evaluation partition")
    entry = next(r for r in panel_manifest() if r["question_id"] == qid)
    if (hashlib.sha256(row["prompt"].encode()).hexdigest() != entry["prompt_sha256"]
            or row["source_row"] != entry["source_row"] or row["source_id"] != entry["source_id"]
            or row["system_prompt"] is not None):
        raise ValueError("Pub97 source question mismatch")
    return [{**row, **item}
            for item in _orders(row["prompt"], spans)]


def source_choice(order, displayed_choice):
    """Map a display label back to its original option; preserve unresolved responses."""
    if order not in ("original", "swapped"):
        raise ValueError("Unknown option order")
    if displayed_choice == "unresolved":
        return "unresolved"
    if displayed_choice not in ("A", "B"):
        raise ValueError("Expected A, B or unresolved")
    if order == "original":
        return displayed_choice
    return {"A": "B", "B": "A"}[displayed_choice]


def extract_choice(presentation, response, extractor):
    """Pass only displayed prompt and answer to an independent semantic extractor."""
    displayed = extractor(presentation["prompt"], response)
    return {"question_id": presentation["question_id"], "order": presentation["order"],
            "displayed_choice": displayed, "source_choice": source_choice(presentation["order"], displayed)}
