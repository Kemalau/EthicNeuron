# Activation anchoring and readouts

Use a population learned on the Pub97 discovery partition. Keep the question IDs, coordinate order, model/tokenizer revisions and checkpoint convention alongside each run. The following snippets use PyTorch and an adapter-equipped model. `selected` is the list of rows emitted by discovery, including zero-based layer and channel IDs and the signed `mean_activation_gap`.

## Activation anchoring

The training driver starts from the original Qwen2.5 or Llama base in the discovery selection and initializes a fresh adapter. Supply your own answers in a UTF-8 JSONL with `question_id` and `response`. Training rows must belong to the Pub97 discovery partition. Every answer for a given ID stays in that partition. Optional held-out answers must belong to its evaluation partition. Preserve the provenance of the supplied answers. Pub97 provides no gold assistant targets.

For an anchored run, provide supervised answers for **a subset** of the discovery questions. Unused discovery IDs form the fixed target-free anchor pool in manifest order, with no evaluation or supervised questions. Use the same supervised file, split and seed across comparison arms. Multiple answer rows for a supervised ID are allowed. This Pub97 construction replaces the paper's separately sourced domain training and anchor pools.

~~~sh
python anchor_training.py --source "$PUB97_SOURCE" --neurons work/population/selected_neurons.json \
  --responses inputs/pub97_train_answers.jsonl --eval-responses inputs/pub97_eval_answers.jsonl \
  --arm selected --seed 0 --beta 1.25 --save-steps 1 --output work/train-selected-s0
~~~

`--arm sft` fits the ordinary supervised baseline. `--arm random --control-seed 42` anchors a single size-matched uniform draw over the entire searched coordinate space excluding the selected neurons (Appendix B.3). Fix the control seed across independently trained seeds. For the sensitivity sweep, use the same inputs with β equal to 0, .05, .1, .25, .5, 1, 1.25 or 2. The coefficient scales the anchor loss once per optimizer update.

The native assistant header and padding receive label −100. Targets include the retained answer and its native end-of-turn marker when it fits within 2,048 tokens. Training runs for one epoch with rank-32 rsLoRA, LoRA alpha 64 and zero dropout on the attention Q/K/V/O and MLP gate/up/down projections. AdamW uses learning rate 0.00001, a five-step linear warmup, then a constant rate. The default microbatch size is two and gradient accumulation is eight. Following the training implementation, each update averages the mean assistant-token losses of its actual microbatches. The final partial group is retained. With a single optimizer step the warmup learning rate for that step is zero.

For each anchored channel, the frozen standard deviation `sigma_anchor` is calculated on all nonpadding prompt tokens in the anchor pool with the adapter disabled (population SD, minimum 0.001). One target-free minibatch per optimizer update contributes `beta × mean(((a_current − a_base)/sigma_anchor)^2)` over valid tokens and selected channels. The accompanying `checkpoints.json` records saved steps, hyperparameters, supervised and anchor question IDs, and the target-file digest. With `--eval-responses` it also records token-weighted held-out NLL at the final checkpoint. Run each seed and arm in a separate output directory.

## Checkpoint trajectory

Measure each checkpoint on identical held-out Pub97 prompts at the final assistant-header token. The original model at step 0 supplies the reference activations. Fix the selected channels, a disjoint control of the same size in each layer, and each group's **own unscaled** signed mean discovery gaps across seeds and steps.

The trajectory scale is computed separately from the anchor-loss scale. The training manifest supplies `calibration_question_ids` in supervised-row order, including repeats. The code computes `sigma_train` on all nonpadding base prompt/template/header tokens of those rows (population SD, minimum 0.001). A custom checkpoint manifest that omits those IDs uses the discovery partition as a calibration proxy and labels that choice in the output. For each group of K channels the scalar is `mean_q dot((a_step(q) − a_base(q))/sigma, unit(mean_gap/sigma)) / sqrt(K)`.

~~~sh
python trajectory.py --source "$PUB97_SOURCE" --neurons work/population/selected_neurons.json \
  --scores work/population/all_scores.npz --checkpoints work/train-selected-s0/checkpoints.json \
  --behavior-draws 2 --output work/trajectory
python judge.py --input work/trajectory/generations.jsonl --output-dir work/trajectory-judge \
  --endpoint "$EM_JUDGE_ENDPOINT" --model "$EM_JUDGE_MODEL"
~~~

`trajectory.json` records the actual training seed and optimizer step for each state measurement. Match its `condition` with the independent judge summary to compare the state with confirmed behavioral EM at the same step. Set `--behavior-draws 0` for state-only measurement. A custom checkpoint manifest uses the discovery `base` and `base_revision`, and a `checkpoints` array of objects containing `seed`, `step` and `adapter`. Paths are relative to that manifest, and remote adapter IDs require an immutable `adapter_revision`. Step 0 is added from the frozen base. Use only observed steps.

## Vocabulary readout

The input is `selected_neurons.json` from this package's Pub97 discovery stage. For each selected channel `i` at layer `l`, its signed gap `D_i` is the adapted-minus-original boundary activation, averaged over paired discovery questions. Section 4.6 and Fig. 5 use the following static readout.

~~~text
v_i = (alpha × D_i) W_down,l[:, i]           alpha = 2 by default
h_G = sum over i in G of P_l v_i
z_G = U RMSNorm_final(h_G)
~~~

The sign of each score describes the projection of the isolated write onto an output token. The command runs on the frozen weights without sampling answers. `--write-weights base` uses the original model's down-projection columns. `--write-weights effective` includes its unmerged LoRA updates at their recorded scaling, composed in FP32 from the saved factors. Both use the original text decoder's native BF16 final norm and output head. Writes and layer transports use FP32 with TF32 disabled. Channels are ordered by layer and channel ID. Layer sums are combined before the single final norm and unembedding.

**Pre-fitted transports.** Supply a local copy of the corresponding file from revision `d2527c3cdbaa60dd4b65cbca1604921f044e989a`. The loader checks the file's SHA-256, the original checkpoint binding, and the dimensions and availability of all requested layers. The fits are external to Pub97. [Pinned Neuronpedia lens files](https://huggingface.co/neuronpedia/jacobian-lens/tree/d2527c3cdbaa60dd4b65cbca1604921f044e989a)

| Original text model | CLI method | File within the pinned release | Fitting prompts |
| --- | --- | --- | ---: |
| Qwen2.5-7B-Instruct | `jacobian` | `qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt` | 485 |
| Llama-3.1-8B-Instruct | `jacobian` | `llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt` | 461 |
| Qwen3.5-9B | `jpp` | `qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt` | 64 |

**Layer selection and comparisons.** By default, the command reads every occupied layer separately and, when several layers are present, their joint write. Add `--per-neuron` to also project each selected channel individually with the chosen lens. For the layer-12/13 grouping, point the following paths at a Llama Pub97 discovery run and the Llama Jacobian file.

~~~sh
python vocabulary_readout.py --neurons work/llama-population/selected_neurons.json \
  --method jacobian --lens "$LLAMA_LENS" --layers 12 13 \
  --scores work/llama-population/all_scores.npz --output work/llama-l12-l13
~~~

The selected groups also get the ordinary logit-lens comparison `P_l = I`. Run `--method logit` alone to compute this comparison without a lens file. Across multiple layers, summing untransported writes serves as a geometric control. The Python interface `methods.readouts.vocabulary_scores` also accepts caller-supplied down-projection matrices and optional layer maps.

**Matched random controls.** Passing `--scores` enables ten draws at seeds 0 through 9 from the full per-layer discovery gap table. Each draw samples without replacement from the unselected channels in that layer, uses their own signed gaps and matches the selected channel count. The code applies one scale to all writes in the draw, so the L2 norm of their raw sum matches the selected raw sum in that layer. It then transports them. Joint controls pair the same seed across layers.

**Optional centred FastICA.** In the 7B environment install `scikit-learn==1.6.1`. Add `--ica-components 4 8 --ica-seeds 0 1 2` to a transported readout with sufficient centred rank. Each observation is one selected channel's dose-weighted, transported write. The fit centres each hidden dimension over observations, preserves their relative magnitudes, and uses parallel FastICA with logcosh, SVD unit-variance whitening, a tolerance of 0.0001 and up to 4,000 iterations. Each mixing-matrix column is oriented towards the uncentred total write and fed to the same vocabulary head after final normalization. The manifest records convergence and the fraction of centred energy reconstructed.

**Outputs.** `scores.npz` stores a full signed score vector per group indexed by token ID, including padded output-head rows. `tokens.csv` lists positive and negative tails. The default takes 50 raw and 30 filtered slots at each end. Tied scores share their original rank (one plus the number of strictly better scores at that end), and token ID orders ties for display. The filtered table preserves the raw rank and removes only special tokens and decoded strings containing no Unicode letter or number. Case, spacing, non-English text and word pieces are preserved. With random controls, `control_ranks.csv` gives each displayed selected token's same-end ranks in all ten controls, the best random rank and the number of controls with rank at most 50. `manifest.json` records the selection, dose, checkpoints, lens, group membership, norm matching and output digests.

## Answer likelihood

`answer_loss_sum(logits, input_ids, target_mask)` returns summed teacher-forced answer-token NLL in nats and the count of retained answer tokens. Provide a boolean mask marking your own answer tokens conditioned on the Pub97 questions, with prompt and padding positions set to false. For each batch, the logits at position `t−1` predict token `t`. Divide the sum of losses across batches by the sum of their token counts. Record the tokenizer, target provenance, EOS rule and sequence cap with the calculation.

## Counterbalanced decisions on Pub97

Pub97 items `extended_000` to `extended_009` present two explicitly labelled résumé alternatives with symmetric instructions. The driver preserves the original question, swaps only the candidate bodies, and uses both presentations once per held-out eligible ID. It applies greedy decoding (256-token limit) to the original base, steered base with α=+2, and unsteered adapted model.

The default seed 42 with 20 reserved questions puts all ten candidates in discovery. For a new study of option order, choose the partition **before discovery**. For example, seed 0 with 20 reserved questions puts one of the ten in evaluation. The choice driver reads the split from the learned selection and stops if it has no eligible held-out questions. Preserve the split for every subsequent step.

~~~sh
python choice_evaluation.py generate --source "$PUB97_SOURCE" \
  --neurons work/choice-population/selected_neurons.json --output work/choice-generations.json
python choice_evaluation.py extract --source "$PUB97_SOURCE" --input work/choice-generations.json \
  --endpoint "$EM_JUDGE_ENDPOINT" --model "$EM_JUDGE_MODEL" --output work/choice-judgments.json
~~~

The independent extractor sees only the displayed question and answer. The summary maps the displayed letter back to the source candidate and reports A/B counts, unresolved responses and consistency across orders. Pub97 provides no normative or risk label for these choices.

## Scope of training comparisons

The core training loop also accepts `--arm kl` and `--arm treft` on the same Pub97 supervised rows. The table states how each comparison is provided. Recreating the original domain-specific comparisons requires their training and evaluation inputs.

| Comparator in §4.8 | Inputs or implementation used by the original comparison |
| --- | --- |
| KL-to-Base | Built in as `--arm kl`. Average reverse KL, D(p_current ∥ p_base), over all attended prompt and response positions at each microbatch. The frozen reference disables the trainable adapter. The coefficient is 0.1. |
| TReFT | Built in as `--arm treft`. Penalize relative squared drift of the key and value projections of shared template-prefix tokens across layers, once per optimizer update. The coefficient is 10. The projection is applied to the input residual of each decoder layer, before head reshaping or rotary embeddings, following the released implementation. [Upstream implementation](https://github.com/CHATS-lab/Token-Regularized-Fine-Tuning/blob/55e6be55404cdbbedb04b3da71ecea230069f10d/release/src/sft.py) |
| BLOCK-EM | Requires a compatible sparse autoencoder and domain-matched latent selection. The manuscript uses layer 15 for Qwen2.5 and 11 for Llama, with coefficient 13,000. [Upstream implementation](https://github.com/ustaomeroglu/block-em/tree/12b983d1ec43d6db62f9de7c2604484995a9ac23) |
| Interleaving++ | Inserts one labelled protection example per 20 supervised rows. [Upstream implementation](https://github.com/davidkaczer/emergent-misalignment/tree/ae9ad39318b3f2133fafbce036545ca0f7507045) |

The paper's domain-task adherence percentage requires domain-specific evaluation prompts and a separate domain judge. Pub97 answer-token likelihood, EM ratings and option choices each measure their stated outcome.
