"""Read frozen Clef exports and restore provenance outside the native model state."""
import json
from pathlib import Path
from .dataset import SPLITS, state_text, validate_record, resolve_data
from .schema import questions
from .storage import file_hash
from .utils import fingerprint


def jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            raise ValueError(f'{path.name}:{number}: Blank dataset line')
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Expected an object')
        except ValueError as error:
            raise ValueError(f'{path.name}:{number}: Invalid JSON object') from error
        rows.append(row)
    if not rows:
        raise ValueError(f'Empty dataset: {path.name}')
    return rows


def source_file(source: str, config: dict) -> Path:
    path = resolve_data(source, config)
    if path.is_dir():
        files = list(path.glob('*.jsonl'))
        if len(files) != 1:
            raise ValueError('Input channel must contain exactly one JSONL file')
        path = files[0]
    return path


def flat_record(row: dict, qs: dict, line: int) -> dict:
    """Retain existing labels verbatim; add only non-model provenance fields."""
    from .data_generation.schemas import validate
    validate(row)
    meta = row['metadata']
    return {
        'id': row['id'], 'master_line': line,
        'group_id': meta['scenario_family_id'],
        'conversation_id': fingerprint(row['customer_context']),
        'leakage_groups': [f'{key}:{meta[key]}' for key in
                           ('scenario_family_id', 'source_template_id', 'response_content_family_id') if meta.get(key)],
        'assigned_split': meta.get('split'),
        'state': state_text(row['customer_context'], row['agent_response']),
        'questions': qs, 'gold': dict(row['labels']),
    }


def load_decision_splits(config: dict) -> dict:
    data = config['dataset']
    if set(data['files']) != set(SPLITS):
        raise ValueError('D1–D6 requires all four explicit split files; automatic resplitting is disabled')
    qs = questions(config)
    manifest = None
    if data.get('manifest_path'):
        path = resolve_data(data['manifest_path'], config)
        if not data.get('manifest_sha256') or file_hash(path) != data['manifest_sha256']:
            raise ValueError('Frozen dataset manifest checksum mismatch')
        manifest = json.loads(path.read_text())
        if manifest.get('dataset_version') != data['version']:
            raise ValueError('Dataset version differs from the frozen manifest')
    provenance = {}
    master_ids = set()
    if data.get('provenance_path'):
        path = source_file(data['provenance_path'], config)
        digest = file_hash(path)
        if not data.get('provenance_sha256') or digest != data['provenance_sha256']:
            raise ValueError('Frozen master provenance checksum mismatch')
        if manifest and manifest['files'].get('master_dataset.jsonl') != digest:
            raise ValueError('Master differs from the frozen manifest')
        for number, row in enumerate(jsonl(path), 1):
            restored = flat_record(row, qs, number)
            key = fingerprint(restored['state'])
            if key in provenance or restored['id'] in master_ids:
                raise ValueError('Ambiguous state or record ID in master provenance')
            provenance[key] = restored
            master_ids.add(restored['id'])
    result = {}
    observed_ids = set()
    for split in SPLITS:
        path = source_file(data['files'][split], config)
        inputs = jsonl(path)
        is_flat = 'customer_context' in inputs[0]
        if manifest:
            relative = f'{"splits" if is_flat else "clef"}/{split}.jsonl'
            if file_hash(path) != manifest['files'].get(relative):
                raise ValueError(f'{split} file checksum differs from frozen {relative}')
        rows = []
        for number, original in enumerate(inputs, 1):
            candidate = flat_record(original, qs, number) if 'customer_context' in original else dict(original)
            validate_record(candidate, data['labels'])
            if candidate['questions'] != qs or list(candidate['questions']) != list(qs):
                raise ValueError(f'{split}:{number}: Questions differ from configured schema')
            if provenance:
                source = provenance.get(fingerprint(candidate['state']))
                if source is None or candidate['gold'] != source['gold']:
                    raise ValueError(f'{split}:{number}: Input or gold differs from master provenance')
                if candidate.get('id', source['id']) != source['id']:
                    raise ValueError('Export record ID differs from master provenance')
                row = dict(source)
            else:
                row = candidate
            if row.get('assigned_split', split) != split:
                raise ValueError('Record is assigned to a different frozen split')
            if not row.get('id') or not (row.get('group_id') or row.get('conversation_id')):
                raise ValueError('Bare Clef rows require master provenance to restore IDs and grouping')
            if row['id'] in observed_ids:
                raise ValueError('Repeated record ID across dataset splits')
            observed_ids.add(row['id'])
            rows.append(row)
        result[split] = rows
    if provenance and observed_ids != master_ids:
        raise ValueError('Split files do not cover the complete master provenance')
    return result
