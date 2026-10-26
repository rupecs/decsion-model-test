# Offline v14 demonstrations

Open **index.html** in a modern browser for 30 real, saved model predictions. Open **dataset.html** for all **17,467 frozen v14 records and their proposed labels**. Both HTML files include their own data, scripts and styling; no installation, internet, server, model, API key or GPU is required. Keep the two files together to use their navigation links. A browser that prohibits local files can instead preview this folder with `python -m http.server 8766 --bind 127.0.0.1 --directory demo` from the repository root.

The prediction gallery contains 10 single-turn cases, 10 multi-turn cases and 10 saved voice snapshots (five two-stage responses). Results were captured from the completed v14 synthetic-development checkpoint on 2026-10-26 UTC. These are model outputs, including errors and contradictions, not ground truth or an independent evaluation. Inputs are newly authored synthetic demonstration cases. The temporary inference GPU was stopped after capture.

The dataset explorer shows proposed synthetic annotations, **not predictions**. It includes train, validation, calibration and test filters. Independent annotation remains pending; no confidence scores are attached to dataset labels. D1–D6 are Boolean decisions, D6 is credit advice, and D7 is the final class.

Use Search, conversation-style and class filters, Previous/Next, or Random to browse. The chooser shows the first 500 matches in the full dataset; search or Next can reach the rest. Where saved response stages exist, the slider switches between actual snapshots and their associated labels/results. It does not generate predictions for arbitrary text. Some complete responses also belong to these voice trajectories.

**Slide view** hides browsing controls. **Export slide PNG** downloads a 1920×1080 image of the selected case and decisions. Long conversations are visibly shortened with an ellipsis in this compact export; the browser view and JSON retain the full input. Check the complete case before presenting it.

`cases.json` contains authored inputs; `predictions.json` preserves the original captured results and input/checkpoint hashes. Do not silently change or repair these outputs. `assets/` contains the editable UI source. Rebuild both HTML files from the repository root:

```sh
python scripts/build_offline_demo.py
```

This is CPU-only and uses already captured predictions. The builder verifies input binding, native decision contracts, the frozen dataset hash and extending snapshot trajectories. Rebuilds do not call an endpoint or change v14. Future captures require a separately authorised inference session; the capture helper is not invoked by the UI or build. See [the runbook](../docs/offline_demos.md).
