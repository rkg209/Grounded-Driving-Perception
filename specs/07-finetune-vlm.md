# Spec 07 — LoRA fine-tune Qwen2.5-VL-3B

**Status:** draft · **Compute:** **cluster only** (SLURM) · **Depends on:** 06

## Objective

LoRA fine-tune **Qwen2.5-VL-3B-Instruct** on the DriveLM train subset, producing the Stage-2 model.
This is the skill transfer the portfolio narrative depends on: *QLoRA on a text LLM → LoRA on a
vision-language model* — the same technique, a new modality.

## Inputs / outputs

- **In:** `Qwen/Qwen2.5-VL-3B-Instruct`, `data/drivelm/train.jsonl` (spec 06).
- **Out:** LoRA adapter (gitignored) + training curves. **Evaluation is spec 08, deliberately
  separate** — so that "did it train?" and "is it better?" cannot blur into each other.

## Approach

1. `transformers` + `peft`. LoRA on the **language-model attention projections**
   (`q,k,v,o,gate,up,down`); **vision tower frozen** by default — the visual encoder is already
   strong and unfreezing it on a small subset invites overfitting. Ablate unfreezing the merger if
   time allows.
2. bf16, gradient checkpointing, `r=16 / α=32` (in `gdp.config.VLMConfig`), AdamW, cosine, warmup.
3. **Mask the loss to answer tokens only.** Computing loss over the question and image tokens
   teaches the model to regurgitate prompts and quietly wrecks accuracy. Assert the label mask in a
   test: everything before the answer must be `-100`.
4. Watch image-token budget: Qwen2.5-VL's dynamic resolution can explode sequence length. Cap
   `max_pixels` and record it — it is a real accuracy/latency lever, not a detail.
5. **Overfit-a-tiny-subset first** (~16 QA pairs → loss ≈ 0) before any full run, same discipline as
   spec 03. Cheap, and it catches every label-masking bug.
6. Emitted as a SLURM script via `/train-job`. Never launched in-session (CLAUDE.md §4).

## Acceptance criteria

1. The 16-sample overfit test drives loss to near-zero.
2. Label-mask test passes: no loss on question/image tokens.
3. A full run completes; adapter saved; curves logged (W&B); no NaNs.
4. The adapter loads back and generates a coherent answer on a fixture image (a *generation* check,
   not an accuracy claim).
5. Training config recorded: subset size, seed, `max_pixels`, LoRA targets, frozen modules.

## Honesty contract

- **H1** — LoRA adaptation of a pretrained VLM. Not "built a VLM", not "trained a VLM".
- **H6** — this model is **entirely separate** from the Stage-1 detector. It does not consume the
  detector's boxes and shares no weights. Never let the demo or write-up imply a joint model.
- **H2** — **no evaluation happens in this spec.** Any number produced here would be a training
  metric, not a result. Resist the urge to peek and report.

## Out of scope

No RLHF/DPO, no full fine-tune, no evaluation (spec 08), no ONNX export of the VLM (spec 05 scope
note), no multi-image temporal reasoning (**H5**).
