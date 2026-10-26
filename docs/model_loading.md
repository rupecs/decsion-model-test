# Original base and portable artifacts

The repository does not contain model weights. Stage the original licensed Google Gemma 4 E4B-IT snapshot locally or through the approved SageMaker File channel. Set DG_MODEL_PATH to its root directory. Preserve all original shards, config, tokenizer and unified processor_config.json unchanged.

Use the normal AWS execution role/provider chain for any permitted staging; do not commit or write access keys into configuration. The SageMaker recipe uses a local base channel and runtime network isolation, so the model integration does not fetch weights from a repository or call an online model API.

`base_identity` checks the full original shard/config identity. A portable checkpoint contains language adapters, the exact FP32 native Clef head, processor, calibration, schema, split assignments, resolved configuration, metadata and a checksum manifest; it omits base weights and intermediate optimiser state.

Both adapter_config.json:base_model_name_or_path and unsloth_decision_config.json:base_model refer to the base. The relocation helper edits these fields only in a disposable metadata copy, verifies the active original base and leaves the checkpoint itself unchanged. Do not directly rewrite a distributed checkpoint's JSON without a new manifest and verification.

After transfer, verify every manifest hash and reload on the actual GPU using the same base. The CLI command is in docs/sagemaker.md. A saved reload fixture can contain synthetic text; do not treat it as independent validation. Never save ordinary confidential inference submissions as fixtures by default.

The exact reference config/shard identity is included as reports/reference_base_identity.json (hashes only, no weights). The AWS native coverage stage compares the supplied original base against it before loading GPU weights and refuses a mismatch.
