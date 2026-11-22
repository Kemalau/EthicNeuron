"""Sample native medical answers for discovery or paired Pub97 held-out evaluation answers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from discovery import decoder_layers, load_model, sha256, token_ids
from steering import SparseSteering

from pub97 import (DEFAULT_EVAL_SIZE, DEFAULT_SEED, MANIFEST_SHA256, SOURCE_SHA256,
                   load_pub97, panel_ids, select_split)

MODEL_FIELDS = ("base", "adapter", "base_revision", "adapter_revision")


def batch_seed(seed, first_pair):
    data = f"pub97-v1|{seed}|{first_pair}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**63-1)


def prefix(tokenizer, task, model_type):
    messages = []
    if task["system_prompt"] is not None:
        messages.append({"role": "system", "content": task["system_prompt"]})
    messages.append({"role": "user", "content": task["prompt"]})
    options = {"enable_thinking": False} if model_type == "qwen3_5" else {}
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return token_ids(ids, "prompt_token_ids")


@torch.inference_mode()
def run_batch(model, tokenizer, selected, batch, recipient, alpha, seed, max_new_tokens, *, greedy=False):
    """Apply one sparse hook per batch and trim each answer at its first EOS."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    if not eos_ids:
        raise ValueError("Recipient has no EOS token ID")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    device = model.get_input_embeddings().weight.device
    model_type = model.get_base_model().config.model_type
    prefixes = [prefix(tokenizer, row, model_type) for row in batch]
    width = max(map(len, prefixes))
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for j, tokens in enumerate(prefixes):
        ids[j, -len(tokens):] = torch.tensor(tokens, device=device)
        mask[j, -len(tokens):] = 1
    torch.manual_seed(seed)
    intervention = (SparseSteering(decoder_layers(model), selected, alpha, mask, decode_steps=32)
                    if selected and alpha != 0 else nullcontext())
    decoding = (dict(do_sample=False) if greedy else
                dict(do_sample=True, temperature=1., top_p=1., top_k=0))
    with model.disable_adapter() if recipient == "base" else nullcontext():
        with intervention as edit:
            output = model.generate(input_ids=ids, attention_mask=mask, **decoding, repetition_penalty=1., use_cache=True,
                                    num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens,
                                    eos_token_id=eos_ids, pad_token_id=pad_id)
    records = []
    for task, prompt_ids, answer in zip(batch, prefixes, output[:, width:].tolist(), strict=True):
        stop = next((i for i, token in enumerate(answer) if token in eos_ids), None)
        answer = answer if stop is None else answer[:stop+1]
        records.append(dict(**task, seed=seed, recipient=recipient, alpha=alpha, prompt_token_ids=prompt_ids,
                            generated_token_ids=answer, response=tokenizer.decode(answer, skip_special_tokens=True),
                            ended_eos=stop is not None, truncated=stop is None,
                            write_counts=edit.edited_calls if edit is not None else {}))
    return records


def model_binding(args, selection):
    """Derive the recipient from the same checkpoint pair used in discovery."""
    fields = {k: getattr(args, k) for k in MODEL_FIELDS}
    if selection is None:
        if not fields["base"] or not fields["adapter"]:
            raise ValueError("Specify --base and --adapter for unsteered discovery")
    else:
        if (selection["panel"], selection["part"]) != ("pub97", "discovery"):
            raise ValueError("Selection requires a Pub97 discovery binding")
        expected_ids = panel_ids("discovery", seed=args.split_seed, eval_size=args.eval_size)
        if (selection["split_seed"] != args.split_seed or selection["eval_size"] != args.eval_size
                or sorted(selection["discovery_question_ids"]) != sorted(expected_ids)):
            raise ValueError("Selection was learned on a different Pub97 partition")
        if selection.get("control_recipient", args.recipient) != args.recipient:
            raise ValueError("Matched control was scaled for a different recipient")
        if not selection["selected"]:
            raise ValueError("Empty selection")
        for k, v in fields.items():
            if v is not None and v != selection[k]:
                raise ValueError(f"Selection checkpoint mismatch: {k}")
            fields[k] = selection[k]
    # Resolve remote refs once so generation and attribution use the same snapshots.
    for key in ["base", "adapter"]:
        path = Path(fields[key]).expanduser()
        if path.is_dir():
            fields[key] = str(path.resolve())
        elif selection is None:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(fields[key], revision=fields[key + "_revision"],
                                         allow_patterns=["config.json", "adapter_config.json"],
                                         local_files_only=args.local_files_only)
            fields[key + "_revision"] = Path(snapshot).name
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["discovery", "evaluation"])
    parser.add_argument("--source", type=Path, required=True, help="Local upstream CSV pinned by Pub97")
    parser.add_argument("--base", help="Original instruction checkpoint directory or model ID")
    parser.add_argument("--adapter", help="Medical adapter directory or model ID")
    parser.add_argument("--base-revision", help="Pinned revision for a remote base")
    parser.add_argument("--adapter-revision", help="Pinned revision for a remote adapter")
    parser.add_argument("--neurons", type=Path, help="Discovery selection or matched control JSON for evaluation")
    parser.add_argument("--recipient", choices=["base", "medical"], default="medical")
    parser.add_argument("--alpha", type=float, default=0.)
    parser.add_argument("--condition", help="Label for an evaluation arm")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, separate from the split seed")
    parser.add_argument("--draws", type=int, help="Samples per question (discovery 10, evaluation 2)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, help="Token cap (discovery 1024, evaluation 600)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New generation JSONL")
    args = parser.parse_args()
    discovery = args.stage == "discovery"
    if discovery and (args.neurons or args.recipient != "medical" or args.alpha):
        parser.error("Discovery requires unsteered answers from the medical recipient")
    if not discovery and (not args.neurons or args.eval_size == 0):
        parser.error("Evaluation requires a selection and held-out questions")
    draws = args.draws if args.draws is not None else (10 if discovery else 2)
    cap = args.max_new_tokens if args.max_new_tokens is not None else (1024 if discovery else 600)
    if min(draws, cap, args.batch_size) < 1 or not math.isfinite(args.alpha):
        parser.error("Sampling sizes must be positive and the dose must be finite")
    if args.output.exists():
        parser.error("Output file already exists")
    questions = select_split(load_pub97(args.source), args.stage, seed=args.split_seed, eval_size=args.eval_size)
    selection = json.loads(args.neurons.read_text(encoding="utf-8")) if args.neurons else None
    model_ids = model_binding(args, selection)
    selected = selection["selected"] if selection else []
    model, tok = load_model(model_ids["base"], model_ids["adapter"], args.device, getattr(torch, args.dtype),
                            base_revision=model_ids["base_revision"], adapter_revision=model_ids["adapter_revision"],
                            local_files_only=args.local_files_only)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    condition = args.condition or ("discovery_medical" if discovery else f"{args.recipient}_alpha{args.alpha:g}")
    membership = "\n".join(q["question_id"] for q in questions).encode()
    binding = dict(model_ids, manifest_sha256=MANIFEST_SHA256, source_sha256=SOURCE_SHA256,
                   panel_sha256=hashlib.sha256(membership).hexdigest(),
                   selection_sha256=sha256(args.neurons) if args.neurons else None,
                   samples=draws, master_seed=args.seed, batch_size=args.batch_size,
                   max_new_tokens=cap, decode_steps=32, temperature=1., top_k=0, top_p=1.,
                   repetition_penalty=1., dtype=args.dtype)
    tasks = [dict(q, sample_idx=i, pair_id=f"pub97:{args.stage}:{q['question_id']}:{i}")
             for q in questions for i in range(draws)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent) as temp:
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start+args.batch_size]
            seed = batch_seed(args.seed, batch[0]["pair_id"])
            for r in run_batch(model, tok, selected, batch, args.recipient, args.alpha, seed, cap):
                r.update(**model_ids, panel="pub97", part=args.stage, split_seed=args.split_seed,
                         eval_size=args.eval_size, id=f"{condition}:{r['pair_id']}", condition=condition,
                         binding=binding)
                temp.write(json.dumps(r, ensure_ascii=False) + "\n")
        temp.flush()
        os.link(temp.name, args.output)
    print(f"Wrote {len(tasks)} Pub97 {args.stage} draws to {args.output}")


if __name__ == "__main__":
    main()
