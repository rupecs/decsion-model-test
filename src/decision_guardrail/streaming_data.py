"""Load immutable D1–D7 exports and restore lineage outside native model state."""
import json
from pathlib import Path

from .dataset import SPLITS, resolve_data, state_text, validate_record
from .schema import D1_D7, questions
from .storage import file_hash, safe_member


def load_streaming_splits(config: dict) -> dict:
    data = config['dataset']
    if set(data['files']) != set(SPLITS):
        raise ValueError('Streaming training requires all four explicit frozen splits')
    manifest_path = resolve_data(data['manifest_path'], config)
    if not data.get('manifest_sha256') or file_hash(manifest_path) != data['manifest_sha256']:
        raise ValueError('Streaming manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != data['schema'] or manifest.get('generator') != data['version']:
        raise ValueError('Frozen streaming version/schema differs from configuration')
    root = manifest_path.parent
    for name, expected in manifest['files'].items():
        if file_hash(safe_member(root, name)) != expected:
            raise ValueError('Frozen streaming artifact checksum mismatch: '+name)
    master = resolve_data(data['provenance_path'], config)
    if file_hash(master) != data['provenance_sha256'] or file_hash(master) != manifest['files']['records.jsonl']:
        raise ValueError('Streaming master provenance checksum mismatch')
    source = [json.loads(line) for line in master.read_text().splitlines()]
    if len(source) != manifest['records'] or len({r['id'] for r in source}) != len(source):
        raise ValueError('Streaming master count or IDs differ from manifest')
    qs = questions(config)
    export_directory = manifest.get('native_export_directory', 'clef')
    if export_directory not in {'clef', 'construction_clef'}:
        raise ValueError('Unknown native export directory')
    if export_directory == 'construction_clef' and (manifest.get('training_ready') is not False or data.get('export_mode') != 'construction_review_only'):
        raise ValueError('Pre-annotation exports require explicit construction-review mode')
    result, seen = {}, set()
    for split in SPLITS:
        path = resolve_data(data['files'][split], config)
        if file_hash(path) != manifest['files'][f'{export_directory}/{split}.jsonl']:
            raise ValueError('Streaming native export checksum mismatch: '+split)
        native = [json.loads(line) for line in path.read_text().splitlines()]
        index = [json.loads(line) for line in safe_member(root,f'clef_index/{split}.jsonl').read_text().splitlines()]
        selected = []
        if len(native) != len(index):
            raise ValueError('Streaming native export and index counts differ')
        for line, (row, ref) in enumerate(zip(native,index,strict=True),1):
            master_line = ref['records_jsonl_line']
            if type(master_line) is not int or not 1 <= master_line <= len(source):
                raise ValueError('Streaming master line outside provenance')
            original = source[master_line-1]
            meta = original['metadata']
            expected = dict(state=state_text(**original['input']), questions=qs, gold=original['labels'])
            validate_record(row, data['labels'])
            if row != expected or list(row['questions']) != list(qs):
                raise ValueError('Streaming state/questions/gold differ from frozen master')
            if ref['export_line'] != line or ref['record_id'] != original['id'] or meta['split'] != split:
                raise ValueError('Streaming native lineage or assigned split differs')
            if original['id'] in seen:
                raise ValueError('Streaming repeated record ID across exports')
            if ref['recommendation_withdrawn_in_target'] != meta['recommendation_withdrawn_in_target']:
                raise ValueError('Streaming withdrawal lineage flag differs')
            seen.add(original['id'])
            selected.append(dict(**row, id=original['id'], master_line=master_line,
                group_id=meta['split_group'], conversation_id=meta['customer_situation_id'],
                leakage_groups=['topic:'+meta['contrast_family']], assigned_split=split,
                trajectory_id=meta['source_refs'][0]['scenario_id'],
                checkpoint_index=meta['source_refs'][0]['checkpoint_index'],
                recommendation_withdrawn_in_target=meta['recommendation_withdrawn_in_target']))
        result[split] = selected
    if seen != {r['id'] for r in source}:
        raise ValueError('Streaming exports do not cover the complete master')
    return result
