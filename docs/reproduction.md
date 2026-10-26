# Reproducing v14 exactly

## What is reproduced

`python scripts/rebuild_v14.py --output .replays/NEW_NAME` regenerates **every frozen release artifact byte-for-byte**, including the original manifest, all four native exports and review files. It refuses an existing output and verifies every result against the original manifest before publishing the new directory. The packaged release remains unchanged.

The successful handover replay reproduced 17,467 rows and all 31 release files. `python scripts/verify_v14.py` separately verifies file hashes, native/master joins, IDs, logical label contracts, dialogue bindings, partition coverage, leakage and streaming trajectories. No Torch, model download or GPU is needed for either command.

## Actual generation inputs

V14 was assembled from two inputs:

1. **12,712 fresh snapshots** derived from `data/authored/streaming_context_v14/products.json` and `services.json`. The generation logic produces real target alternatives and prefixes, preserves role-labelled histories, merges identical input rows and tracks trajectories. It uses the original helper functions for label profiles, ASR surfaces, IDs and JSON serialization.
2. **4,755 exact retained v14 members** in `retained_v14_members.json.gz`. These records originated in v13, but every record in this seed is already a member of the final v14 dataset. No unused v13 row or complete older dataset is bundled. Their original IDs, labels, splits, design identities and provenance remain unchanged.

`portable_v14.py` preserves the original assembly logic. Its historical input boundary loads the pinned retained-v14 seed rather than the original workspace's v13 release and correction working pool. It also preserves the original ancestor manifest hash as provenance, without claiming an absent older release was re-verified. The original assembly script and exact original source snapshot are in `provenance/` for inspection.

The portable assembler has been checked against the published master: all 17,467 dictionaries, 356 designs and dialogue bindings match exactly. It does not merely copy `records.jsonl` into the replay directory.

## Packaging and historical evidence

The original publisher depended on earlier reviews, older datasets, a v12 review-page template and workstation output folders. Those do not belong in this public handover. The portable replay instead includes:

- a review-page **shell** with its old dataset payload removed; v14 data is regenerated into it;
- pinned frozen release metadata: original manifest, quality report, correction ledger and historical README;
- the retained-v14 seed described above;
- original source code and source hashes for audit.

Large master records, native exports, split/index files, review payloads, blank annotation packets, cue reports and trajectories are regenerated. Historical audit metadata is preserved as historical evidence. This is **not** a new independent annotation or a claim that the private multi-version review workspace was reconstructed.

The original release README deliberately retains old workspace commands and publication-era “pending” flags, because changing it would change the release checksum. Use this document and the root README for the supported portable workflow.

## Frozen authoring and future changes

`provenance/authoring_pins.json` locks the authoring inputs and replay metadata. Changing an authored line, seed, template or expected output causes replay to fail. That is intentional: this tool reproduces v14; it is not a publisher for silently modified v14 content.

For a future corrected corpus, create a new version, rerun exposure/grounding/coherence/shortcut/coverage audits, obtain independent annotation and update the complete contract deliberately. Do not change expected hashes merely to make a failing v14 replay pass. Publishing this synthetic corpus makes its held-out data public; create a separately authored independent challenge set for future claims.

## Installation and checks

Use Python 3.12 for parity with the tested GPU runtime. CPU-only checks also ran locally without Torch. From the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/verify_package.py
python scripts/verify_v14.py
python -m pytest -q
python scripts/rebuild_v14.py --output .replays/v14-check
python scripts/review_public_tree.py
```

Keep `.replays/`, generated job JSON and future output/model artifacts out of Git. No old release folder or workstation path is required.

## Source integrity versus historical source hashes

The frozen release manifest’s source_hashes describe the original publication workspace, including historical paths intentionally absent from this public repository. They are not claims that the portable/AWS adaptations have the same source hashes. The original source snapshot preserves the corresponding generation code; package_manifest.json separately binds the current public code/data/docs. reports/reference_runtime_core_hashes.json verifies that the critical model/trainer/calibration/saving/reload and early-stopping modules remain byte-identical to the actually tested A40 runtime. The changed boundaries are the default public profile, explicit portable opt-in gate, authoring seed loader and new SageMaker/replay orchestration. A fresh actual-GPU smoke/reload remains required.

The supported deployment is a source checkout or the supplied Docker build. Installing a wheel alone does not install the large frozen dataset or its authoring inputs.
