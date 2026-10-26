# V14 data and reference-model card

## Intended use

Research/development classification of the current assistant response in a banking/financial conversation, including partial voice text. D1–D6 are supporting Boolean decisions and D7 is the final four-class output. This is not a financial advice generator, legal approval system or validated automated decision tool for customers.

## Data origin and composition

All examples are synthetic authored proposals. V14 contains 17,467 snapshots, 356 designs and 1,291 distinct histories across single-turn, multi-turn and partial-response trajectories. All four partitions retain original assignments and grouping. Each partition covers the required product/service domains and call categories; structural coverage is not proof of semantic diversity or realistic prevalence.

Class counts across the complete corpus:

| D7 label | Snapshots |
|---|---:|
| other | 5,705 |
| factual_information | 6,054 |
| personal_advice | 3,506 |
| general_advice | 2,202 |

4,755 exact existing members were retained into v14, and 12,712 new snapshots were assembled from v14 authored inputs. Only v14 members are supplied; no complete earlier dataset is included. The original master/manifest hashes and per-row lineage are preserved.

## Annotation and review status

The manifest remains pre_annotation_not_trainable with training_ready=false, approved_training_records=0 and an empty independent annotation ledger. Proposed labels and historical construction/model-review evidence are included to support handover and adjudication. They do not replace independent annotation. The known A01 grounding issue and later model-review exposure are documented rather than hidden.

## Reference model and results

The original BF16 Gemma 4 E4B-IT base stayed frozen, with language LoRA and a new native FP32 Clef head. The A40 development run passed all-row token coverage, real three-step smoke, exact FP32 saving and all-seven fresh-process offline reload. It stopped at 600 optimiser steps and restored step 300 before calibration. The raw reference metrics and validation history are in reports/reference_training_metrics.json.

Synthetic held-out D7 accuracy: 0.9887880406; macro-F1: 0.9853063929; all-seven exact match: 0.8974906567; consistent fraction: 0.9839829151 (30 inconsistent records). D1 macro-F1: 0.7898794908. These results are not independent challenge, real-call, legal or SageMaker validation.

## Exclusions and required prerequisites

Base/adapter/head weights, raw machine/run connection files, credentials and old datasets are not published. The receiving operator must supply the original licensed base, an approved AWS environment, fresh actual-GPU smoke/reload and explicit synthetic-development authorisation. Read docs/known_issues.md and AGENTS.md before use.
