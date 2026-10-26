"""Build a review-only context contrast batch; never produce training exports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy

from decision_guardrail.dataset import state_text, split_state
from decision_guardrail.schema import decision_violations

VERSION = 'banking_streaming_d1_d7_v5_context_review_draft'
STATUS = 'construction_proposal_pending_independent_annotation'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def construct(root: Path) -> tuple[list[dict], list[dict]]:
    content = runpy.run_path(str(root / 'data/authored/streaming_context_v5/seed_families.py'))
    profiles, families = content['PROFILES'], content['FAMILIES']
    rows = []
    specs = []
    for family in families:
        family_id = 'context-v5-' + family['id']
        ids = []
        for index, (context, target, profile, rationale) in enumerate(family['rows'], 1):
            canonical_context, _ = split_state(state_text(context, target))
            turns = [{'id': f'context-{i}', 'text': text} for i, text in enumerate(canonical_context.splitlines(), 1)]
            bits = profiles[profile]
            labels = {f'D{i}': bool(value) for i, value in enumerate(bits, 1)}
            labels['D7'] = ('personal_advice' if labels['D3'] or labels['D4'] else 'general_advice') if labels['D2'] else ('factual_information' if labels['D5'] else 'other')
            row = dict(id=f'{family_id}-{index:02}', schema_version=VERSION,
                input=dict(customer_context=canonical_context, agent_response=target),
                proposed_labels=labels,
                annotation=dict(status=STATUS, rule_version=VERSION, author='Codex construction author',
                    independent_reviewer=None, adjudicated=False, profile=profile, rationale=rationale,
                    context_turns=turns,
                    evidence={d: dict(target_spans=[dict(start=0,end=len(target),text=target)],
                        context_turn_ids=[t['id'] for t in turns] if d in {'D1','D3','D4','D6'} else [])
                        for d in labels if d != 'D7' and labels[d]}),
                metadata=dict(family_id=family_id, purpose=family['purpose'], domain=family['domain'],
                    variant_index=index, split=None, training_eligible=False,
                    exposed_development=True, independent_challenge=False,
                    source='data/authored/streaming_context_v5/seed_families.py',
                    source_family=family['id']))
            row['review_fingerprint'] = digest(row)
            rows.append(row)
            ids.append(row['id'])
        specs.append(dict(id=family_id, purpose=family['purpose'], domain=family['domain'],
            design=family['design'], row_ids=ids, split=None,
            lineage_note='Compositional development construction; not independently authored or independently adjudicated.'))
    return rows, specs


def validate(rows: list[dict], families: list[dict]) -> dict:
    if len(rows) != 200 or len(families) != 50:
        raise ValueError('Review milestone requires 200 distinct rows across 50 families')
    seen = set()
    keys = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate ID')
        seen.add(row['id'])
        content_key = digest(row['input'])
        if content_key in keys:
            raise ValueError('Duplicate full input')
        keys.add(content_key)
        if row['schema_version'] != VERSION or set(row['proposed_labels']) != {f'D{i}' for i in range(1,8)}:
            raise ValueError('Unexpected schema')
        labels = row['proposed_labels']
        if any(type(labels[f'D{i}']) is not bool for i in range(1,7)) or decision_violations(labels):
            raise ValueError('Invalid decision proposals')
        if row['metadata']['training_eligible'] is not False or row['metadata']['split'] is not None:
            raise ValueError('Review batch cannot be exported for training')
        if row['annotation']['status'] != STATUS or row['annotation']['adjudicated']:
            raise ValueError('Construction proposals cannot claim adjudication')
        payload = {k:v for k,v in row.items() if k != 'review_fingerprint'}
        if row['review_fingerprint'] != digest(payload):
            raise ValueError('Stale review fingerprint')
        if split_state(state_text(**row['input'])) != tuple(row['input'][k] for k in ('customer_context','agent_response')):
            raise ValueError('Target round-trip changed')
        for evidence in row['annotation']['evidence'].values():
            if not evidence['target_spans']:
                raise ValueError('Missing positive evidence')
            for span in evidence['target_spans']:
                if row['input']['agent_response'][span['start']:span['end']] != span['text']:
                    raise ValueError('Invalid evidence span')
            if not set(evidence['context_turn_ids']) <= {t['id'] for t in row['annotation']['context_turns']}:
                raise ValueError('Unknown evidence turn')
    grouped = {family['id']: [row for row in rows if row['metadata']['family_id'] == family['id']] for family in families}
    if sum(map(len,grouped.values())) != len(rows):
        raise ValueError('Unknown family')
    for family in families:
        members = grouped[family['id']]
        if [r['id'] for r in members] != family['row_ids'] or len(members) != 4:
            raise ValueError('Family members differ')
        if family['purpose'] == 'prefix':
            if len({r['input']['customer_context'] for r in members}) != 1:
                raise ValueError('Prefix history differs')
            for earlier,later in zip(members,members[1:]):
                if not later['input']['agent_response'].startswith(earlier['input']['agent_response']):
                    raise ValueError('Snapshot includes a non-prefix edit')
        if family['purpose'] in {'implicit_context','credit'}:
            if len({r['input']['agent_response'] for r in members[:3]}) != 1:
                raise ValueError('Context swap changes target')
            if [r['proposed_labels']['D7'] for r in members] != ['general_advice','personal_advice','general_advice','factual_information']:
                raise ValueError('Context contrast proposals differ from design')
    return dict(records=len(rows), families=len(families),
        class_counts=dict(Counter(r['proposed_labels']['D7'] for r in rows)),
        family_purposes=dict(Counter(f['purpose'] for f in families)),
        domains=dict(Counter(r['metadata']['domain'] for r in rows)),
        independent_annotations=0, training_eligible_records=0,
        token_coverage='pending actual native encoder; CPU structure checks are not token coverage')


def build(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Output already exists; choose a new path to preserve artifacts')
    rows, families = construct(root)
    report = validate(rows,families)
    template = (root/'scripts/templates/context_v5_review.html').read_text()
    payload = json.dumps(rows,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    output.mkdir(parents=True)
    write_rows(output/'records.jsonl',rows)
    write_json(output/'families.json',families)
    write_json(output/'quality_report.json',report)
    (output/'review.html').write_text(template.replace('__REVIEW_DATA__',payload),encoding='utf-8')
    (output/'README.md').write_text('''# Context v5: 200-snapshot development review batch

200 construction proposals in 50 compositional families; not independent gold.
No training, validation, calibration or test assignment. No native training exports.
Open review.html locally. Labels are hidden initially for an independent first pass.
Enter reviewer identity, assign D1–D7 and a rationale, and export JSONL comments.
Comments stay in this browser until exported; they do not alter these artifacts.
Review keys include a full-record fingerprint. Exported labels are judgments, not
an automatic adjudication. Disputes must be resolved and exclusions recorded.

The author has seen the development probes. This is not a fresh blind challenge.
Repeated structures deliberately support matched contrasts; 200 rows are not
200 independent scenarios. Broad lexical diversity and native token fitness remain
unverified. Evidence spans currently cover the whole short target; reviewers should
record narrower spans or turn references when resolving a boundary.
''',encoding='utf-8')
    sources=['data/authored/streaming_context_v5/seed_families.py',
        'docs/annotation_standard_streaming_v5.md','config/dataset/question_schema_streaming_v5_review.yaml',
        'src/decision_guardrail/data_generation/context_v5_review.py','src/decision_guardrail/dataset.py',
        'scripts/templates/context_v5_review.html']
    write_json(output/'manifest.json',dict(schema_version=VERSION,status=STATUS,
        source_sha256={p:sha(root/p) for p in sources},
        files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return report


def verify(output: Path) -> dict:
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest.get('schema_version') != VERSION or manifest.get('status') != STATUS:
        raise ValueError('Unexpected manifest schema/status')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(output/name) != expected:
            raise ValueError('Artifact hash mismatch')
    return validate([json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()],
                    json.loads((output/'families.json').read_text()))
