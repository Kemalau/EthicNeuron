#!/usr/bin/env python3
"""Train fresh rsLoRA adapters on caller-supplied answers to identified Pub97 questions."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from discovery import decoder_layers, sha256, token_ids
from generate import prefix
from methods.activations import ActivationAnchor, SelectedChannels
from methods.readouts import answer_loss_sum
from methods.training_comparisons import SharedPrefixKV, kl_to_base
from pub97 import SOURCE_SHA256, load_pub97, panel_ids, selected_layers


def read_responses(path, eligible):
    """Bind each supplied target to an actual Pub97 question in the chosen partition."""
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid, answer = row["question_id"], row["response"]
        if qid not in eligible or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Each target requires an eligible Pub97 ID and a nonempty response")
        if "prompt" in row and row["prompt"] != eligible[qid]["prompt"]:
            raise ValueError("Supplied prompt differs from its Pub97 question")
        examples.append((eligible[qid], answer))
    if not examples:
        raise ValueError("No supervised targets supplied")
    return examples


def encode_target(tok, row, answer, cap):
    """Mask the native assistant header and retain the assistant body and its turn terminator."""
    prompt = prefix(tok, row, "text")
    messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": answer}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = token_ids(list(ids), "chat token IDs")
    if ids[:len(prompt)] != prompt:
        raise ValueError("Assistant target does not follow the same tokenized generation prefix")
    # Retain the first EOS/turn terminator. Drop any template suffix after it.
    suffix = ids[len(prompt):]
    eos = tok.eos_token_id
    if eos in suffix:
        suffix = suffix[:suffix.index(eos) + 1]
    if not suffix or suffix[0] == eos or len(prompt) >= cap:
        raise ValueError("No assistant content remains under the sequence cap")
    ids = (prompt + suffix)[:cap]
    return ids, [-100] * len(prompt) + ids[len(prompt):]


def batch_tokens(examples, pad, device):
    """Right-pad tokens, masking all padding and the supplied prompt prefixes."""
    width = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for j, (tokens, target) in enumerate(examples):
        size = len(tokens)
        ids[j, :size] = ids.new_tensor(tokens)
        mask[j, :size] = 1
        if target is not None:
            labels[j, :size] = ids.new_tensor(target)
    return dict(input_ids=ids, attention_mask=mask, labels=labels)


def random_anchor(model, selection, seed):
    """Appendix B.3 draws once over the searched coordinates, excluding all selected channels."""
    rng = np.random.default_rng(seed)
    layers = decoder_layers(model)
    chosen = {(r["layer"], r["neuron"]) for r in selection["selected"]}
    search = selection["layers"]
    if not search or search != sorted(set(search)) or any(type(l) is not int or not 0 <= l < len(layers) for l in search):
        raise ValueError("Invalid discovery search layers")
    widths = [layers[l].mlp.down_proj.in_features for l in search]
    offsets = np.cumsum([0] + widths)
    positions = {l: n for n, l in enumerate(search)}
    if any(l not in positions or not 0 <= i < widths[positions[l]] for l, i in chosen):
        raise ValueError("Population outside discovery search range")
    excluded = [offsets[positions[l]] + i for l, i in chosen]
    pool = np.setdiff1d(np.arange(offsets[-1]), excluded)
    sampled = rng.choice(pool, len(chosen), replace=False)
    members = []
    for flat in sorted(sampled.tolist()):
        pos = int(np.searchsorted(offsets, flat, side="right") - 1)
        members.append(dict(layer=search[pos], neuron=int(flat - offsets[pos])))
    return members


def fresh_model(base, revision, device, local):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    options = dict(revision=revision, local_files_only=local)
    conf = AutoConfig.from_pretrained(base, **options)
    if conf.model_type not in ("qwen2", "llama"):
        raise ValueError("The paper's training-time anchoring workflow uses Qwen2.5 or Llama")
    tok = AutoTokenizer.from_pretrained(base, **options)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, **options)
    adapter = LoraConfig(r=32, lora_alpha=64, lora_dropout=0., use_rslora=True, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")
    return get_peft_model(model, adapter).to(device), tok


def train(args, out):
    from transformers import set_seed

    selection = json.loads(args.neurons.read_text(encoding="utf-8"))
    selected_layers(selection, None)
    split = dict(seed=selection["split_seed"], eval_size=selection["eval_size"])
    ids = set(panel_ids("discovery", **split))
    questions = load_pub97(args.source)
    train_rows = {q["question_id"]: q for q in questions if q["question_id"] in ids}
    heldout = {q["question_id"]: q for q in questions if q["question_id"] not in ids}
    pairs = read_responses(args.responses, train_rows)
    eval_pairs = read_responses(args.eval_responses, heldout) if args.eval_responses else []
    supervised_ids = [q["question_id"] for q, _ in pairs]
    used_ids = set(supervised_ids)
    anchor_ids = [q for q in train_rows if q not in used_ids]
    beta = args.beta if args.beta is not None else (1.25 if args.arm in ("selected", "random") else 0.)
    if beta and not anchor_ids:
        raise ValueError("Reserve some discovery-partition Pub97 IDs for target-free anchors")
    set_seed(args.seed)
    model, tok = fresh_model(selection["base"], selection["base_revision"], args.device, args.local_files_only)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_examples = [encode_target(tok, q, answer, 2048) for q, answer in pairs]
    eval_examples = [encode_target(tok, q, answer, 2048) for q, answer in eval_pairs]
    template_penalty = None
    if args.arm == "treft":
        template_penalty = SharedPrefixKV(model, tok, [prefix(tok, train_rows[q], "text") for q in used_ids])
    anchor_batches, members, anchor = [], [], None
    if beta:
        members = (selection["selected"] if args.arm == "selected"
                   else random_anchor(model, selection, args.control_seed))
        projections = [l.mlp.down_proj for l in decoder_layers(model)]
        channels = SelectedChannels(projections, members)
        prompts = [(prefix(tok, train_rows[q], "text")[:2048], None) for q in anchor_ids]
        for i in range(0, len(prompts), args.batch_size):
            batch = batch_tokens(prompts[i:i+args.batch_size], pad, "cpu")
            anchor_batches.append({k: batch[k] for k in ("input_ids", "attention_mask")})
        anchor = ActivationAnchor(model, channels, anchor_batches)

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(step / 5., 1.))
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed)).tolist()
    micro = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    total = math.ceil(len(micro) / args.accumulate)
    if any(s > total for s in args.save_steps):
        raise ValueError(f"Requested checkpoint beyond this one-epoch run's {total} optimizer steps")
    checkpoints = []
    for k in range(0, len(micro), args.accumulate):
        block = micro[k:k+args.accumulate]
        step = k // args.accumulate + 1
        optimizer.zero_grad(set_to_none=True)
        if anchor is not None:
            (beta * anchor.loss(anchor_batches[(step-1) % len(anchor_batches)])).backward()
        if template_penalty is not None:
            (10. * template_penalty.loss()).backward()
        for idx in block:
            batch = batch_tokens([train_examples[i] for i in idx], pad, args.device)
            result = model(**batch, use_cache=False)
            loss = result.loss
            if args.arm == "kl":
                loss = loss + 0.1 * kl_to_base(model, batch, result.logits)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            (loss / len(block)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step in args.save_steps or step == total:
            name = f"checkpoint-{step}"
            model.save_pretrained(out / name)
            checkpoints.append(dict(seed=args.seed, step=step, adapter=name))

    manifest = dict(panel="pub97", source_sha256=SOURCE_SHA256, selection_sha256=sha256(args.neurons),
                    base=selection["base"], base_revision=selection["base_revision"], arm=args.arm,
                    beta=beta, seed=args.seed, split_seed=split["seed"], eval_size=split["eval_size"],
                    calibration_question_ids=supervised_ids, anchor_question_ids=anchor_ids if beta else [],
                    anchor_members=[{k:r[k] for k in ("layer","neuron")} for r in members],
                    control_seed=args.control_seed if args.arm == "random" else None,
                    kl_coefficient=0.1 if args.arm == "kl" else 0.,
                    treft_coefficient=10. if args.arm == "treft" else 0.,
                    responses_sha256=sha256(args.responses), batch_size=args.batch_size,
                    accumulation=args.accumulate, epochs=1, optimizer_steps=total,
                    optimizer="AdamW", learning_rate=1e-5, weight_decay=0., gradient_clip_norm=1.,
                    warmup_steps=5, schedule="constant_with_warmup", sequence_cap=2048,
                    rslora_rank=32, lora_alpha=64, lora_dropout=0.,
                    sft_reduction="mean assistant-token loss per microbatch, mean over this accumulation group",
                    checkpoints=checkpoints)
    if eval_examples:
        model.eval()
        nats, tokens = 0., 0
        with torch.no_grad():
            for j in range(0, len(eval_examples), args.batch_size):
                batch = batch_tokens(eval_examples[j:j+args.batch_size], pad, args.device)
                mask = batch.pop("labels") != -100
                logits = model(**batch, use_cache=False).logits
                loss_sum, count = answer_loss_sum(logits, batch["input_ids"], mask)
                nats += float(loss_sum)
                tokens += count
        manifest["heldout"] = dict(responses_sha256=sha256(args.eval_responses),
                                   question_ids=[q["question_id"] for q, _ in eval_pairs], responses=len(eval_pairs),
                                   target_tokens=tokens, total_nats=nats, nats_per_target_token=nats/tokens)
    (out / "checkpoints.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--neurons", type=Path, required=True, help="Frozen Pub97 discovery population")
    parser.add_argument("--responses", type=Path, required=True, help="Supervised JSONL: question_id and response")
    parser.add_argument("--eval-responses", type=Path, help="Compute token-weighted NLL on supplied held-out targets")
    parser.add_argument("--arm", choices=["sft", "selected", "random", "kl", "treft"], required=True)
    parser.add_argument("--beta", type=float, help="Anchor strength (defaults to 1.25 for selected and random)")
    parser.add_argument("--seed", type=int, required=True, help="Adapter initialization and training order")
    parser.add_argument("--control-seed", type=int, default=42, help="Fixed independent random anchor draw")
    parser.add_argument("--save-steps", type=int, nargs="*", default=[], help="Intermediate steps; final is always saved")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.beta is not None and (not math.isfinite(args.beta) or args.beta < 0
                                  or (args.arm not in ("selected", "random") and args.beta)):
        parser.error("A positive anchor beta applies only to the selected and random arms")
    if (args.output.exists() or args.batch_size < 1 or args.accumulate < 1 or not 0 <= args.seed < 2**32
            or args.control_seed < 0 or any(s < 1 for s in args.save_steps)):
        parser.error("Choose a new output directory, positive batch sizes and valid seeds/checkpoint steps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "training"
        staged.mkdir()
        train(args, staged)
        staged.rename(args.output)
    print(f"Saved adapters and checkpoint manifest to {args.output}")


if __name__ == "__main__":
    main()
