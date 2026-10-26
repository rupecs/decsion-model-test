"""CPU-only dataset integration report; never claims tokenizer/GPU validation."""
from collections import Counter
from pathlib import Path
from .dataset import load_splits, split_manifest
from .schema import questions, contract, contract_hash
from .utils import fingerprint, write_json


def prepare_dataset(config: dict) -> dict:
    destination = Path(config['paths']['report_dir'])/'dataset-preparation'
    if destination.exists():
        raise FileExistsError('Preparation report exists; choose a new paths.report_dir')
    splits = load_splits(config)
    assignments = split_manifest(splits)
    coverage = {}
    for split, rows in splits.items():
        coverage[split] = {
            'records': len(rows),
            'gold_counts': {q: dict(Counter(str(row['gold'][q]).lower() if type(row['gold'][q]) is bool else row['gold'][q] for row in rows)) for q in questions(config)},
            'scenario_groups': len({row.get('group_id') for row in rows}),
            'input_sha256': fingerprint([row['state'] for row in rows]),
        }
    report = {
        'status': 'cpu_checks_passed', 'dataset_version': config['dataset']['version'],
        'records': sum(len(rows) for rows in splits.values()),
        'schema': contract(config), 'schema_sha256': contract_hash(config),
        'splits': coverage, 'split_manifest_sha256': fingerprint(assignments),
        'checks': ['question/gold types and consistency', 'exact configured split counts',
                   'unique IDs and normalized states', 'disjoint declared groups',
                   'frozen file checksums and master join when configured'],
        'model_input_fields': ['state', 'questions', 'gold'],
        'token_length_check': {'status': 'not_run', 'reason': 'Requires the staged Gemma tokenizer and pinned native encoder; training rejects overlong or skipped records.'},
        'native_gpu_check': 'not_run',
        'label_status': 'Synthetic construction labels; CPU checks do not certify legal correctness.',
        'artifacts': str(destination),
    }
    from .training_readiness import training_readiness
    report['training_readiness'] = training_readiness(config)
    destination.mkdir(parents=True)
    write_json(destination/'report.json', report)
    write_json(destination/'split_assignments.json', assignments)
    write_json(destination/'resolved_config.json', config)
    return report
