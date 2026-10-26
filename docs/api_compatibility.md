# Native API compatibility

The reference runtime validated Unsloth 2026.10.6 and its corresponding Zoo release on Python 3.12 / Torch 2.8.0+cu128. `environment.inspect_api()` checks the installed exact version and required signatures before model integration. Do not replace signature checks with a successful import alone.

Native integration uses FastDecisionModel.from_pretrained with decision_head='clef', head_init=None for a new FP32 native head; get_peft_model attaches only language adapters. The original base remains frozen. build_dataset and the official encode_record are used for training/token coverage, and native calibrate/evaluate/predict remain the model implementation. Boolean questions use native type `noul` and real Boolean labels; probability keys are `true`/`false`.

Native probabilities are rounded to four decimal places. Validation permits the corresponding rounding tolerance without renormalising or inventing probabilities. A 99% probability accompanying “No” is probability of the selected negative answer.

Exact FP32 serving artifacts use the **private** native `_save_clef_adapter(..., exact=True)` path. Public save_pretrained can reduce head precision or fold calibration, so it is not an interchangeable replacement. Exact version/signature checks and actual save/reload comparisons guard this portability risk.

The AWS entrypoint reuses the reference trainer and the CPU-tested best-state callback, with only a new explicit portable development gate / orchestration boundary. CPU tests do not validate a different AWS GPU stack. Run fresh smoke/save/all-seven reload on the actual hardware, and capture the resulting full freeze and image digest.
