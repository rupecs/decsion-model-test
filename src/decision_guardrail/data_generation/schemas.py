"""Versioned flat master records; all labels concern the separate agent_response."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
CLASSES = ('factual_information', 'general_advice', 'personal_advice', 'other')
VERSION = 'financial_advice_v1'


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def questions():
    return read_yaml(ROOT / 'config/dataset/question_schema_v1.yaml')


def validate(row):
    if row.get('schema_version') != VERSION:
        raise ValueError('Unknown schema version')
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('Missing ID')
    for field in ('customer_context', 'agent_response'):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f'Nonempty {field} required')
    lab = row.get('labels', {})
    if set(lab) != {f'D{i}' for i in range(1,7)}:
        raise ValueError('Six labels required')
    if any(type(lab[f'D{i}']) is not bool for i in range(1,6)) or lab['D6'] not in CLASSES:
        raise ValueError('Invalid decision value')
    if lab['D2'] and (not lab['D1'] or lab['D5']):
        raise ValueError('Product scope or factual-only contradiction')
    if not lab['D2'] and (lab['D3'] or lab['D4']):
        raise ValueError('Not-applicable personalisation must be false')
    expected = ('personal_advice' if lab['D3'] or lab['D4'] else 'general_advice') if lab['D2'] else ('factual_information' if lab['D5'] else 'other')
    if lab['D6'] != expected:
        raise ValueError('Direct D6 contradicts atomic labels; do not rewrite')
    meta = row.get('metadata')
    if not isinstance(meta, dict):
        raise ValueError('Missing metadata')
    for key in ('scenario_family_id', 'source_template_id', 'call_category', 'sub_intent', 'annotation_status'):
        if not meta.get(key):
            raise ValueError(f'Missing metadata {key}')
    for item in row.get('evidence', {}).values():
        text = row[item['field']]
        if text[item['start']:item['end']] != item['text'] or not item['text']:
            raise ValueError('Evidence span mismatch')
    return row


def load_rows(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'master_dataset.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError('Empty dataset')
    seen = set()
    for row in rows:
        validate(row)
        if row['id'] in seen:
            raise ValueError('Duplicate ID')
        seen.add(row['id'])
    return rows
