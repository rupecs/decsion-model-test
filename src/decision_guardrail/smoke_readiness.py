"""Explicit, hash-bound integration fixture exception; never annotation approval.

Only the frozen 20-record D1–D6 fixture may use this path. Changing data, schema
or the bounded execution contract requires a new reviewed exception in code.
"""
from pathlib import Path
from .schema import contract_hash
from .storage import file_hash

FIXTURE_ID = 'd1_d6_20_record_smoke_v1'
CONTRACT_SHA256 = 'c17c148632b3f2d2b6c7eedfdd8e02c0e24e2f7e1ee5fb20fe8e21b3ee889e20'
SPLIT_SHA256 = {
    'train': 'bd06d107da28b613523727d669ff4350a0c034bc8249adf9f73b4dccc3593e74',
    'validation': '3fba046657264d62d5649bec315d9c44dff887df98ef2978d8feda0f4e7323b2',
    'calibration': 'ee5f8e2f2a68ebb1b259afb0896a6562789bf17d5186465bb17a2a857dcdfa7b',
    'test': 'def6fc6225c1da906a005bd687da15c8d836c53064048bc487ee8d2ca2efc156',
}
SPLIT_COUNTS = {'train': 12, 'validation': 3, 'calibration': 2, 'test': 3}


def fixture_smoke_readiness(config: dict) -> dict:
    """CPU-only checks, called inside the fail-closed annotation gate wrapper."""
    data, training = config['dataset'], config['training']
    denied = dict(ready=False, status='fixture_smoke_contract_mismatch', approved_records=0)
    if training.get('fixture_smoke') != FIXTURE_ID:
        return denied
    steps = training.get('max_steps')
    if (config.get('experiment') != 'smoke_test'
        or type(steps) is not int or not 1 <= steps <= 3
        or type(training.get('epochs')) is not int or training['epochs'] != 1
        or training.get('head_only') is not False
        or type(training.get('batch_size')) is not int or training['batch_size'] != 1
        or type(training.get('gradient_accumulation_steps')) is not int or training['gradient_accumulation_steps'] != 8
        or config['model'].get('compile') is not False
        or config['model'].get('max_seq_length') != 2048
        or data.get('schema') != 'd1_d6_v1'
        or data.get('version') != 'synthetic-d1-d6-smoke-v1'
        or data.get('path') is not None
        or data.get('multi_question') is not False
        or data.get('expected_total') != 20
        or data.get('splits') != SPLIT_COUNTS
        or set(data.get('files', {})) != set(SPLIT_SHA256)
        or contract_hash(config) != CONTRACT_SHA256):
        return denied
    for split, expected in SPLIT_SHA256.items():
        source = data['files'][split]
        # Do not stage remote objects or import a GPU stack to check this fixture.
        if not isinstance(source, str) or '://' in source or file_hash(Path(source)) != expected:
            return dict(ready=False, status='fixture_smoke_checksum_mismatch', approved_records=0)
    return dict(ready=True, status='hash_pinned_integration_smoke_only', approved_records=0,
                fixture=FIXTURE_ID, max_optimizer_steps=3,
                reason='Integration testing on the exact frozen synthetic fixture only; no annotation or corpus-training approval.')
