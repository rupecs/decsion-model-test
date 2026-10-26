"""Default-deny annotation gate with an explicit pinned integration fixture path."""
import json
import hashlib
from pathlib import Path
from .storage import file_hash, safe_member


def annotation_fingerprint(record: dict) -> str:
    """Match the review-queue convention, including non-ASCII conversation text."""
    payload=dict(input=record['input'],labels=record['labels'])
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def training_readiness(config: dict) -> dict:
    # No schema or profile name is an exemption. The separate, explicit smoke
    # opt-in is bounded by pinned fixture and question-contract hashes.
    try:
        return _verified_readiness(config)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return dict(ready=False,status='annotation_evidence_unreadable_or_invalid')


def _verified_readiness(config: dict) -> dict:
    data=config['dataset']
    if config['training'].get('development_run') is not None or config['training'].get('fixture_smoke')=='v14_d1_d7_integration_20261026':
        from .development_readiness import check
        return check(config,smoke=config['training'].get('fixture_smoke') is not None)
    if config.get('training',{}).get('fixture_smoke') is not None:
        from .smoke_readiness import fixture_smoke_readiness
        return fixture_smoke_readiness(config)
    path=data.get('manifest_path')
    if not path or str(path).startswith('s3://'):
        return dict(ready=False,status='annotation_manifest_must_be_verified_locally_before_training')
    path=Path(path)
    root=path.parent
    if not data.get('manifest_sha256') or file_hash(path)!=data['manifest_sha256']:
        return dict(ready=False,status='annotation_manifest_checksum_mismatch')
    manifest=json.loads(Path(path).read_text())
    if manifest.get('schema_version') != data.get('schema'):
        return dict(ready=False,status='annotation_manifest_schema_mismatch')
    if manifest.get('training_ready') is not True or manifest.get('status')!='independently_adjudicated_training_release' or manifest.get('native_export_directory','clef')!='clef':
        return dict(ready=False,status='pre_annotation_not_trainable',approved_records=0,
            reason='The verified release manifest has not approved a complete independently adjudicated corpus. Construction exports are for CPU preparation/review only.')
    ledger_name=manifest.get('annotation_ledger_path')
    if not ledger_name or ledger_name not in manifest.get('files',{}):return dict(ready=False,status='missing_annotation_ledger')
    ledger_path=safe_member(root,ledger_name)
    if file_hash(ledger_path)!=manifest['files'][ledger_name]:return dict(ready=False,status='annotation_ledger_checksum_mismatch')
    master_path=safe_member(root,'records.jsonl')
    if file_hash(master_path)!=manifest['files'].get('records.jsonl'):return dict(ready=False,status='annotation_master_checksum_mismatch')
    master=[json.loads(l) for l in master_path.read_text().splitlines()]
    if not master or len({r['id'] for r in master}) != len(master):
        return dict(ready=False,status='empty_or_duplicate_annotation_master')
    if manifest.get('records') != len(master):
        return dict(ready=False,status='annotation_manifest_count_mismatch')
    ledger=json.loads(ledger_path.read_text())
    if not isinstance(ledger,list):return dict(ready=False,status='invalid_annotation_ledger')
    by_id={r.get('record_id'):r for r in ledger}
    if len(by_id)!=len(ledger) or set(by_id)!={r['id'] for r in master}:return dict(ready=False,status='incomplete_annotation_coverage')
    for r in master:
        review=by_id[r['id']]
        if review.get('status')!='adjudicated' or review.get('independent_review') is not True or not review.get('reviewer') or not review.get('author') or review['reviewer']==review['author']:
            return dict(ready=False,status='incomplete_independent_adjudication')
        # The fingerprint convention is explicit and excludes changing metadata.
        if review.get('input_labels_sha256')!=annotation_fingerprint(r):
            return dict(ready=False,status='stale_annotation_fingerprint')
    return dict(ready=True,status='independently_adjudicated_training_release',approved_records=len(master))


def assert_training_ready(config: dict) -> None:
    report=training_readiness(config)
    if not report['ready']:
        raise ValueError('Training blocked by annotation readiness: '+report['status'])
