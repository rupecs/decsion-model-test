# Native training and evaluation contract

## Reference development run

The completed reference run used the original frozen BF16 Gemma 4 E4B-IT backbone, language-layer LoRA and a newly initialised **native Unsloth Clef head in FP32**. Compilation was disabled. The source-level architecture must remain unchanged when porting to AWS.

The reference A40 run passed official encoding of all 17,467 snapshots, a three-step D1–D7 smoke, actual head/LoRA gradient and update checks, exact FP32 saved-head verification and fresh-process offline reload of all seven decisions. It then trained with the fixed cap below. No independent annotation approval was granted.

| Setting | Value |
|---|---|
| Sequence limit | 2,048, including all questions |
| Batch / accumulation | 1 / 8 |
| Seed | 3407 |
| LoRA rank / alpha / dropout | 16 / 16 / 0.0 |
| Learning rate | 0.0002; native head LR default preserved |
| Epoch configuration / optimiser cap | 2 / 2,972 |
| Validation / logs | every 100 / every 5 steps |
| Early stopping | validation loss; patience 3; min_delta 0.001 |
| Best state | exact original-dtype CPU copies of trainable tensors, restored before calibration |
| Precision | BF16 base; native FP32 Clef head |
| Compilation | false |

The cap is computed from the fixed train size and accumulation; reaching the cap is not workflow completion. Calibration, portable save, reload and one held-out test must follow. The reference run stopped at step 600, restored step 300 (validation loss 0.0552177802), and used about 16.95 GiB peak allocated training GPU memory. This is evidence for that A40 stack, not guaranteed memory or performance on every AWS instance.

## Explicit development opt-in

The manifest stays `pre_annotation_not_trainable`, with zero independently approved rows. The normal gate denies it. The versioned development exception requires the exact data/manifest/native export hashes, fixed question contract, bounded settings, supported actual compute profile and explicit operator opt-in.

A CPU-only inspection is available:

```sh
python scripts/sagemaker_train.py --compute l40s_48gb \
  --authorize-synthetic-development --check-config-only
```

This does not load model weights or allocate a GPU. Running the same entrypoint without `--check-config-only` is real GPU work and requires the operator's cloud authorisation and staged original model.

## Per-job prerequisites

Every AWS job runs all-snapshot native coverage and a fresh smoke on its actual GPU. An A40 certificate cannot stand in for an L40S or A100 certificate. The smoke uses a separate, exactly pinned 20-record integration fixture, derived exclusively from v14 train topics; its internal validation/calibration/test subsets are **not** the full release held-out partitions.

The native official encoder checks the complete schema-inclusive sequence. Oversized, skipped or truncated records fail the job. There is no silent row omission or shorter-sequence fallback. The smoke must demonstrate finite loss, head and language-LoRA gradients/updates, a frozen backbone, restored best state, all FP32 saved head tensors, manifest hashes and all-seven fresh-process offline reload. The full profile also requires the matching smoke key.

## Evaluation discipline

Select the model using validation only. Restore the exact best trainable parameters before temperature calibration on calibration only. Preserve the native calibration state in the checkpoint and verify it on reload. Run the held-out test CLI once after final saving/reload; do not tune against its results or silently correct contradictory decisions.

Report D1–D7 separately, especially credit D6 and weaker D1; include D7 accuracy/macro-F1, all-seven exact match and logical consistency. Keep calibration and validation metrics distinguishable. Record losses, selected step, duration, memory, environment/freeze, base identity, dataset/schema hashes and actual cleanup state.

## Failures, stops and portability

There is no training resume implementation. A failed prerequisite blocks full training. There is no automatic restart, quantisation, head substitution or framework/data change. Partial logs are useful evidence, not a complete trained model.

Native exact saving relies on the pinned private `_save_clef_adapter(..., exact=True)` API. After any related integration change, inspect installed signatures and rerun real save/reload on the actual GPU. Keep both base relocation fields intact in portable metadata. Original base weights stay outside model outputs; verify their full identity at the receiving machine.

See [the SageMaker runbook](sagemaker.md), [model loading](model_loading.md) and [known issues](known_issues.md).

Dataset replay is byte-exact. Reusing the training protocol and seed on a different GPU does not guarantee bit-identical trained tensors or identical selected steps/metrics; record the actual AWS result rather than forcing it to match the reference.
