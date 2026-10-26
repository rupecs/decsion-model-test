"""Explicit versioned exposure policies for current audits; frozen loaders stay unchanged."""
import json
from pathlib import Path

from .dialogue_exposure import sha


def discovered_reviews(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for pattern in ('record_review.jsonl', 'blind_sample_review.jsonl')
            for p in (root / 'data/reviews').rglob(pattern)}


def load_policy(root: Path, policy_path: Path | None = None) -> dict:
    root = root.resolve()
    if policy_path is None:
        pointer = root / 'data/CURRENT_STREAMING_DATASET.json'
        name = json.loads(pointer.read_text()).get('cumulative_exposure_policy') if pointer.exists() else None
        policy_path = Path(name or 'config/dataset/dialogue_exposure_policy.json')
    path = policy_path if policy_path.is_absolute() else root / policy_path
    if not path.resolve().is_relative_to(root):
        raise ValueError('Exposure policy must be inside the project')
    policy = json.loads(path.read_text())
    ledger_path = (root / policy['ledger']).resolve()
    if not ledger_path.is_relative_to(root) or sha(ledger_path) != policy['sha256']:
        raise ValueError('Exposure ledger hash mismatch')
    ledger = json.loads(ledger_path.read_text())
    if ledger.get('format') != 'cumulative_dialogue_exposure_v1' or not ledger.get('reviewed_record_ids'):
        raise ValueError('Invalid or empty exposure ledger')
    if set(ledger['review_files']) != discovered_reviews(root):
        raise ValueError('Exposure policy omits a recorded or blind-sample review; renew the ledger')
    for name, expected in {**ledger['review_files'], **ledger['source_files'],
                           **ledger.get('review_id_lists', {})}.items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or sha(source) != expected:
            raise ValueError('Exposure source hash mismatch: ' + name)
    if ledger.get('unresolved_review_ids'):
        raise ValueError('Exposure ledger contains unresolved IDs')
    return {**ledger, 'ledger_path': str(ledger_path.relative_to(root)), 'ledger_sha256': sha(ledger_path),
            'policy_path': str(path.relative_to(root))}
