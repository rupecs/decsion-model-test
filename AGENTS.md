# Instructions for the receiving agent

## Mission and scope

Maintain and reproduce the frozen **v14** synthetic D1–D7 dataset and its native Clef development training workflow in the operator's AWS SageMaker environment. Start by reading README.md, docs/reproduction.md, docs/training.md, docs/sagemaker.md, docs/known_issues.md and docs/annotation_standard_streaming_v14.md.

This repository is a clean handover, not the original multi-version workstation. No old release or private workspace is needed for supported commands. Some internal helper filenames have historical version numbers because they are dependencies of the original generator. They are not invitations to regenerate or train older corpora. `provenance/original_v14_source_snapshot.zip` preserves the exact original source inventory for audit; original workspace commands in that archive are reference-only.

## Immutable data contract

- Release root: `data/generated/streaming_context_v14_release`.
- Manifest SHA256: `90baf0e0e75c4c3331fe89c9d7b3f5c7e3427d6155078e990501efe322f3c281`.
- Master records SHA256: `dd682e928aa3b266e0f2fead96fcc810572b095a4d1d189229556a4b1d1a89c7`.
- 17,467 snapshots: 11,883 train, 1,640 validation, 2,071 calibration, 1,873 test; 356 designs and 1,291 distinct customer histories.
- The internal question schema is `banking_streaming_d1_d7_v6_context_draft`. Dataset version is `streaming-context-dialogue-v14`. Do not rename either to make their numerals match.
- Preserve IDs, labels, split assignments, grouping, authoring provenance, ordering, canonical state text, question ordering and native export bytes.
- Do not randomly re-split, deduplicate, rephrase, relabel, shuffle histories across splits, turn prefixes into full replies, or add examples to v14 in place.
- New annotations, corrections or new training data require a new version and new manifests. Never silently “fix” the frozen data or its checksum files.
- Native exports remain under `construction_clef/`, not `clef/`. The annotation ledger is empty; `training_ready` remains false and independently approved count remains zero.

Run `python scripts/verify_v14.py` before any training preparation. Run `python scripts/rebuild_v14.py --output .replays/NEW_NAME` to reproduce all release files exactly. It does not require older datasets. The retained seed contains only 4,755 rows already present in v14. Historical audit metadata is preserved evidence, not a current independent annotation or a claim of newly re-running all earlier reviews. Consult docs/reproduction.md before altering replay code.

## Meaning of the decisions

Classify the current AGENT response using the complete CUSTOMER/AGENT conversation. Prior agent opinions and customer quotations are context, not automatically a current endorsement. CUSTOMER/AGENT roles must remain explicit. ASSISTANT is only an accepted UI alias normalised by state_text.

- D1: banking / financial product in the target's scope.
- D2: product recommendation or influential product opinion.
- D3: actual customer objectives, financial situation or needs demonstrably used in that advice. The mere presence of a personal disclosure, a comparison activity or an information request is not sufficient.
- D4: individual suitability asserted or implied, including a contextually directed selection without “you”. A bare disclosure by itself is insufficient, but a product choice answering that disclosure can imply fit under this dataset convention.
- D5: substantive facts or procedural help without product advice. A mixed factual/recommendation reply has advice; the earlier factual prefix can have a different label.
- D6: advice about a credit product. This is Boolean; it is not the old six-decision final class.
- D7: factual_information, general_advice, personal_advice or other.

For this contract, D2 plus either D3 or D4 yields personal advice. A generic-request disclaimer does not undo an actual recommendation. Quoted, rejected or fictional opinions must be judged by the actual speech act; no phrase, pronoun or percentage is a ground-truth label. A displayed 99% for “No” is probability of the chosen negative answer, not a positive score.

Partial voice snapshots must be judged only from the text available at that point. Do not inherit an endorsement from words that occur later. Preserve source_refs and trajectory order. These conventions are research annotation rules, not legal determinations.

## Runtime invariants

Use the original, licensed Google Gemma 4 E4B-IT local checkpoint. Preserve its files, unified processor_config.json and full base identity hashes. Never substitute another model, download a different model at runtime, quantise, merge adapters into the base, full-finetune the base, use an LM prompt classifier or introduce a conventional linear head.

Required stack: BF16 frozen base, language-layer LoRA, newly initialised native Unsloth Clef head in FP32, compile=false, maximum sequence length 2048, batch 1 / accumulation 8. The complete tested dependency constraints are in tested_gpu_constraints.txt. The reference run used Python 3.12, Torch 2.8.0+cu128, Unsloth and Unsloth Zoo 2026.10.6, Transformers 5.17.0, PEFT 0.21.2, Triton 3.4.0 and Xformers 0.0.32.post2.

Keep GPU imports lazy. Call set_offline/configure_runtime before GPU imports, and import Unsloth before Transformers/PEFT. Run environment.inspect_api against the installed exact stack. Never silently resolve a failure by changing framework, precision, schema, data, compilation or hardware profile. The private exact FP32 saving API is version/signature guarded; reload verification is mandatory after any related change.

Default training is denied. A separate operator must knowingly pass `--authorize-synthetic-development` to the bounded workflow. That opt-in does not approve labels independently and does not authorise this agent to spend cloud money without the operator's request. Do not set `training_ready: true`, broaden exemptions, forge a smoke certificate, or interpret a copied reference certificate as proof on AWS.

## SageMaker execution

Use `sagemaker/Dockerfile` and `scripts/sagemaker_train.py`. The fixed dataset and smoke fixture are baked into the image; only the licensed base model arrives through the `base_model` File input channel. See docs/sagemaker.md for why the four dataset channels must not override these pinned baked paths.

Before submitting, confirm the AWS account/region, approved training role, private S3 model/output prefixes, pinned ECR image digest, actual GPU instance availability/quota, VPC settings and any KMS grants with the operator. Never invent these values or embed credentials. Use the normal AWS provider chain / execution role. Job rendering is CPU-only and never submits. Keep generated account-specific job JSON under ignored `.local/`.

Choose a compute profile that matches the actual GPU (L40S 48GB, A100 80GB, or reference A40 48GB). Do not assume an EC2 instance family is available for SageMaker training in a particular region. Expose one GPU only. A multi-GPU instance with CUDA_VISIBLE_DEVICES=0 still bills the full instance; do not choose one casually. Small-VRAM fallback is unsupported.

The single job must execute, in order:

1. CPU integrity and native master/export joins.
2. Exact official native encoding of all 17,467 snapshots, including all seven questions; zero skipped or truncated rows.
3. Fresh three-step D1–D7 smoke on the actual AWS GPU with head/LoRA gradients and updates.
4. Exact saved-head FP32 tensor/header verification and fresh-process offline reload of all seven decisions, including invalid saved-base-path portability testing.
5. Full development training only after those checks pass. Cap 2,972 optimiser steps, epoch configuration 2, evaluation every 100 and logging every 5.
6. Validation-loss early stopping: patience 3, min_delta 0.001. Preserve original tensor dtype, restore exact best trainable tensors before calibration, and never copy or modify frozen base tensors.
7. Calibration on calibration only; portable saving and full fresh-process reload.
8. Exactly one held-out test CLI evaluation; never use test results to select a model or tune the run.

The workflow has one-shot stage locks and no automatic restarts or resume. Do not rerun a failed stage into the same output path. Investigate saved logs and obtain a new run decision. SageMaker MaxRuntime is 12 hours; on-demand is the default. SIGTERM recovery/resume is not implemented. Do not use spot interruptions as if recovery were supported.

Write only final portable artifacts under /opt/ml/model, and logs/reports/freeze/phase/exit under /opt/ml/output/data. The original base stays in its channel. Intermediate optimiser/trainer and duplicate smoke weights must not be included in the final portable model. On completion/failure/StopTrainingJob, verify terminal SageMaker status and that no separate endpoint or resource was left running. Retained S3/ECR storage is separate from training compute.

## Validation and honest reporting

CPU checks: `python -m pytest -q`, `python scripts/verify_v14.py`, and a replay to a new directory. `python scripts/sagemaker_train.py --compute l40s_48gb --authorize-synthetic-development --check-config-only` verifies the opt-in/bounds without importing GPU libraries.

The reference A40 run passed coverage, three-step smoke, FP32 saving and all-seven reload. It stopped at step 600 and restored step 300. Its synthetic D7 test accuracy was 0.988788, macro-F1 0.985306 and all-seven exact match 0.897491. D1 macro-F1 was 0.789879. Read reports/reference_training_metrics.json; do not present these as AWS or independent challenge results.

Known limitations include A01 grounding on shortened histories, residual lexical/template cues, unsolicited suitability controls, public/model-review exposure and a basic factual inference probe where D5 and D7 disagreed. Zero independently approved rows. Independent annotation/adjudication and a truly independent challenge set remain outstanding. A successful transport/API call is not proof that a predicted label is correct. Preserve raw native outputs and flag contradictions; never “repair” them silently.

## Public repository hygiene

Do not commit base/adapter/head weights, checkpoints, original workstation outputs, logs containing submitted customer text, SSH keys/known_hosts, credential files, tokens, account-specific job files, private bucket names, office material, notebooks with secrets or .venv/cache/build artifacts. Normal model inputs must not be logged by default. No project content goes to feedback or telemetry services.

Before any push, inspect the exact Git diff and staged file inventory; run scripts/review_public_tree.py and inspect every finding. Never push the original workstation folder wholesale. Review compressed seeds/source snapshots too. Preserve an existing user-selected destination license; do not invent a new license or imply model redistribution rights.

Only a new explicit instruction authorises a paid AWS job or deployment. This handover does not launch cloud work. Do not recreate any Runpod resource. Keep changes focused, add meaningful CPU tests for new data/gate/workflow behaviour, and explain what was and was not actually executed.

## Offline demos

`demo/index.html` contains 30 genuine saved v14 model outputs; `demo/dataset.html` contains all frozen dataset labels. Preserve the distinction: dataset annotations have no model confidence, and predictions are not ground truth. All demonstration inputs are authored synthetic cases. Never publish normal customer submissions. Preserve `demo/predictions.json` and its case/checkpoint binding; do not repair model errors. Rebuild with `python scripts/build_offline_demo.py` and run the CPU tests after UI edits. The HTML is self-contained with network connections disabled by CSP. Browsing and rebuilding must never launch inference, training or cloud resources. A future capture requires explicit operator authorisation; the temporary GPU used for the supplied results was stopped.
