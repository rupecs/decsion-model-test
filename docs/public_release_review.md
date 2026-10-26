# Public release selection and review

## Included

- The complete **unchanged v14** release: 17,467 snapshots, original manifest, master, four native exports, split/index files, schema, proposed annotation material and historical construction diagnostics.
- Calibration and test partitions as well as train/validation, because they are necessary to reproduce the completed workflow without contaminating validation or fitting calibration on test.
- Current v14 authoring inputs and a compressed seed containing only the 4,755 retained members already present in v14.
- Portable generation/review code and its transitive helper dependencies; exact original source/build scripts are retained separately as provenance. Historically named helper modules are code, not old released datasets. Unused older generator modules were removed from the active source tree.
- The 20-row v14-derived integration fixture, reference base **hashes only**, sanitised reference metrics, exact tested package constraints and the relevant native runtime.
- Explicit default-deny synthetic-development opt-in, CPU tests, bounded AWS workflow, Dockerfile, review-only job renderer, AGENTS.md and sectioned runbooks.

## Excluded

- Every complete older dataset release, abandoned candidate, old review batch, private multi-version working pool and original workstation `outputs/` tree.
- Original base weights, adapters, Clef head weights, optimiser/intermediate checkpoints and trained-model archives.
- SSH keys, known_hosts, live connection/session files, Runpod IDs/addresses, private job/account configuration, AWS credentials, tokens and local environments/caches.
- Normal inference input logs, screenshots/browser state, notebooks with execution state, private office content and unrelated projects.
- Raw cloud billing/account records and cloud access helpers. No old Git history is copied from the workstation.

## Checks performed for this handover

- Verified the original v14 master and manifest hashes and all native/master joins, grouping, label contracts and all four counts.
- Regenerated all 31 release files byte-for-byte. A separate clean replay also succeeded without the packaged release directory or any older dataset, demonstrating that the authoring inputs are sufficient.
- Ran focused CPU tests for the default-deny gate, exact data/step/precision bounds, early stopping, exact-dtype best-state restoration, SageMaker request generation, CPU-only opt-in checks and publication screening.
- Reviewed the selected runtime/configuration/docs, file inventory and reference-data provenance. Kept the dataset's annotation status and historical audit limits explicit.
- Screened plain files and decompressed source/retained inputs for common token/key formats, home-directory paths, live SSH endpoints, non-loopback IPv4 addresses, forbidden operational/model files and the GitHub 100-MiB single-file limit. Initial matches were reviewed: four-part pinned package versions and the scanner's own detector text were false positives; scanner-specific tests guard those cases.
- No AWS training job, endpoint, S3 upload or image push was launched during preparation. No public GitHub push is implied by these checks.

## Limits and owner decisions

Local pattern screening and manual inspection do not prove that arbitrary future files contain no confidential information; rerun review on the exact staged diff before every push, including compressed additions. A destination repository must also be inspected for existing files/history and the owner's chosen license before integrating this package.

No software/data license was invented. Keep or explicitly choose the destination terms. Original model acquisition/licensing remains the receiving operator's prerequisite. No third-party model weights or downloaded package source are bundled.

The SageMaker adaptation is CPU-tested. Docker image building, account/region-specific IAM/quota/instance settings and actual AWS GPU smoke/reload remain receiving-environment work. Synthetic development labels are not production approval, independent challenge validation or legal validation.

## Recorded result

44 focused CPU tests passed; the rendered job request also matches the installed AWS service schema without making an API call. The clean replay reproduced all 31 frozen release files without the packaged release or any older dataset. Local screening found no remaining credential/operational-file findings after reviewed false positives; compressed inputs were inspected too. The largest file is 49,120,555 bytes, below GitHub’s 100-MiB per-file limit. The destination rupecs/decsion-model-test was confirmed public and empty, with no existing license selected. No model weights or old Git history are imported. See reports/handover_verification.json and the complete package_manifest.json.

## Offline demo addition

Added self-contained prediction and dataset-label HTML pages with editable UI sources and CPU-only rebuilding. The prediction page preserves 30 genuine captured outputs on newly authored synthetic demonstration inputs, including model errors; no customer submissions were captured. The dataset page joins all 17,467 frozen record inputs, labels and partitions without attaching confidence scores. Network access is disabled by the page CSP; no cloud connection, resource ID, credentials or model weights are embedded.

This later, explicitly authorised inference-only capture used a temporary A40 session. It did not train a model or launch AWS resources. The GPU was stopped and read back as EXITED after capture, and the checkpoint manifest remained unchanged. Existing model/base storage was preserved separately. Local GPU connection files and logs remain excluded from the public repository.

The expanded CPU suite passed 53 tests. A deterministic offline rebuild, complete source-to-display joins and saved-stage checks passed. Browser checks confirmed the two modes, dataset partition filtering, actual voice-stage navigation and a 1920×1080 PNG export. The browser preview used localhost; direct file navigation was not available in the automated browser. Both HTML files contain all their assets and data, with no fetch or external dependency.
