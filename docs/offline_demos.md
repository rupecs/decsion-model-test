# Offline demonstration runbook

The handover includes two separate self-contained pages:

- `demo/index.html`: 30 genuine captured predictions from the completed v14 synthetic-development checkpoint. Ten single-turn cases, ten multi-turn cases and five two-stage voice families. All outputs, including errors, remain unchanged.
- `demo/dataset.html`: all 17,467 frozen v14 records with their proposed D1–D7 labels and original partitions. No model probability is assigned to these labels.

Clone/download the repository, then open either HTML file in a modern browser. Keep both together for navigation. There are no external assets, network calls, API keys or GPU requirements. GitHub's file viewer displays HTML source; download first. If local-file navigation is disabled by the browser, preview only the demo directory on loopback:

```sh
python -m http.server 8766 --bind 127.0.0.1 --directory demo
```

Browse [http://127.0.0.1:8766/](http://127.0.0.1:8766/). This serves saved files only, not inference. Use an unused port if one is occupied. See [the user guide](../demo/README.md) for filtering, saved voice stages and slide export. Long inputs are visibly shortened in the compact PNG; full text remains in the browser and source JSON.

## Reproduce the pages

From the repository root:

```sh
python scripts/build_offline_demo.py
python -m pytest -q tests/test_offline_demo.py
```

The builder is CPU-only. It binds each recorded result to its authored case, validates all seven native answers, checks the frozen master hash and verifies snapshot extension. It embeds all data/assets into each generated HTML file, and neither changes the frozen dataset nor contacts inference. `--output NEW_DIRECTORY` can produce a comparison copy.

## Capture provenance and limits

`demo/cases.json` is the authored synthetic input batch. `demo/predictions.json` records raw native outputs, capture timestamps, actual model version, checkpoint-manifest SHA256 and cases SHA256. Captures were made on 2026-10-26 UTC using inference only; the temporary GPU was stopped after all 30 requests, and base/checkpoint storage was preserved. This is a demonstration batch, not an independent challenge benchmark. The original training data remains synthetic and independently unapproved.

The capture helper `scripts/capture_demo_predictions.py` is separate from both the UI and builder. It accepts only a loopback HTTP endpoint for an already authorised model session, verifies the v14 model identity, requires the verified checkpoint-manifest hash, and refuses to overwrite an existing capture. It does not provision or start a model, GPU or endpoint. Future capture needs explicit operator authorisation, a new output file and fresh review; never capture ordinary customer submissions for publication. There are no model weights or cloud connection details in this demo.

Do not modify confidence scores, replace errors with expected labels, interpolate predictions between saved stages or suggest that proposed dataset labels were model outputs. A corrected capture or altered input batch must be a distinct, documented version.
