# V14 banking decision dataset and SageMaker handover

This repository contains the **exact v14 dataset used in the completed development run**, its reproducible authoring inputs and generation scripts, and a single-GPU SageMaker training workflow. It contains no older dataset releases, model weights, credentials or live cloud connection details.

| Partition | Snapshots | Purpose |
|---|---:|---|
| Train | 11,883 | Optimisation |
| Validation | 1,640 | Early stopping / model selection |
| Calibration | 2,071 | Native temperature calibration |
| Test | 1,873 | One final held-out evaluation |
| **Total** | **17,467** | 356 designs; 1,291 distinct customer histories |

All four partitions are included because reproducing this model requires separate calibration and test data as well as training and validation. The corpus mixes single-turn, multi-turn and partial voice responses. D1–D6 are Boolean supporting decisions; **D6 means credit advice**, and **D7 is the final four-class label**. The internal schema is named `banking_streaming_d1_d7_v6_context_draft`; that schema name is intentionally different from the dataset release number.

**These are frozen synthetic labels, not independently approved production labels.** The completed run was explicitly a bounded development experiment. The manifest remains `training_ready: false` with zero independently approved records. Training requires an explicit synthetic-development opt-in; do not change the manifest to bypass this distinction.

## Start here

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/verify_package.py
python scripts/verify_v14.py
python -m pytest -q
python scripts/rebuild_v14.py --output .replays/v14-first-rebuild
```

These commands are CPU-only. The rebuild regenerates all 31 frozen release files byte-for-byte without needing older datasets. It refuses changed authoring inputs, differing output bytes and an existing output directory. See [the precise reproduction boundary](docs/reproduction.md): 12,712 snapshots are assembled from v14 authored designs; 4,755 exact retained members are included as v14 inputs, not as an older release.

## Train in your AWS environment

Follow [the SageMaker runbook](docs/sagemaker.md), which covers the original base-model prerequisite, an ECR image, account/region-specific resource selection, IAM/VPC/KMS inputs, a reviewable job request, execution, logs, outputs and shutdown. **No script in this repository submits an AWS job automatically.** The actual AWS GPU must pass fresh native token coverage and a three-step D1–D7 smoke/save/offline-reload test before full training.

The workflow preserves the original BF16 Gemma 4 E4B-IT base, language LoRA, native FP32 Clef head and `compile=false`. It caps full optimisation at 2,972 steps, validates every 100 steps, logs every 5, and restores the exact validation-best trainable tensors before calibration. See [runtime details](docs/training.md).

The receiving agent should read [AGENTS.md](AGENTS.md) before editing or launching anything.

## Reference results and limitations

The A40 development run stopped early at step 600 and restored step 300. Synthetic held-out D7 accuracy was 98.8788%, macro-F1 0.985306; all-seven exact match was 89.7491%. D1 macro-F1 was only 0.789879. These are shared-recipe synthetic results, **not independent challenge validation or legal approval**. AWS execution has not been claimed tested. See [the model/data card](docs/data_and_model_card.md), [reference metrics](reports/reference_training_metrics.json) and [known weaknesses](docs/known_issues.md).

## Repository map

- `data/generated/streaming_context_v14_release/`: complete frozen v14 release; full records, native exports, split/index files, review material and original manifest.
- `data/authored/streaming_context_v14/`: v14 designs, exact retained-v14 seed and review-page shell.
- `data/samples/v14_development_smoke/`: the 20-record integration fixture, derived only from the v14 train partition.
- `scripts/`: portable replay, data verification, GPU coverage and bounded SageMaker workflow.
- `src/decision_guardrail/`: native model integration and the generator's transitive helper dependencies. Some helper filenames carry historical version numbers; there are no older dataset releases here.
- `sagemaker/`: Dockerfile and a job-request renderer that never calls AWS.
- `provenance/`: original generation scripts/source snapshot, frozen historical audit metadata and authoring hashes.
- `docs/`: reproduction, annotation, runtime, AWS and publication guidance.

The README inside the frozen release is a historical artifact and retains its original publication-era “pending” statements and old workspace command. Use the commands above and the current runbooks; do not edit frozen files to modernise their wording.

See [public release review](docs/public_release_review.md) and [NOTICE](NOTICE.md) before publishing. No new software or dataset license has been selected by this handover; preserve any license explicitly chosen in the destination repository.
