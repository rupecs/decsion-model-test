"""Separately versioned downstream policy, independent of classification."""
from pathlib import Path
import yaml


def load_policy(path: str | Path) -> dict:
    """Read and validate actions and confidence thresholds."""
    policy=yaml.safe_load(Path(path).read_text())
    if not policy.get('version') or policy.get('default_action') not in {'allow','review','block'}:
        raise ValueError('Policy requires version and a valid default action')
    for rule in policy.get('rules',{}).values():
        if rule.get('action') not in {'allow','review','block'} or not 0<=rule.get('minimum_confidence',0)<=1:
            raise ValueError('Invalid policy rule')
    return policy


def apply_policy(classification: dict, policy: dict) -> dict:
    """Return a policy decision without altering model probabilities."""
    # D6 is the native final classification. Atomic predictions are not silently
    # combined into a new class or treated as an implicit enforcement policy.
    if 'decisions' in classification:
        final = 'D7' if 'D7' in classification['decisions'] else 'D6'
        answer = classification['decisions'][final]
        label, probabilities = answer['answer'], answer['probabilities']
    else:
        label, probabilities = classification['advice_type'], classification['probabilities']
    rule=policy.get('rules',{}).get(label)
    action=policy['default_action']
    if rule and probabilities[label]>=rule.get('minimum_confidence',0):
        action=rule['action']
    return {'action':action,'policy_version':policy['version']}
