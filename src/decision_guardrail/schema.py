"""Versioned decision contracts shared by training, serving and checkpoint reload."""
from copy import deepcopy
from .utils import fingerprint

LEGACY = 'legacy_v1'
D1_D6 = 'd1_d6_v1'
D1_D7 = 'banking_streaming_d1_d7_v4_binary_draft'
D1_D7_CONTEXT = 'banking_streaming_d1_d7_v5_context_draft'
D1_D7_CONTEXT_V6 = 'banking_streaming_d1_d7_v6_context_draft'
STREAMING_SCHEMAS = (D1_D7, D1_D7_CONTEXT, D1_D7_CONTEXT_V6)
DECISIONS = tuple(f'D{i}' for i in range(1, 7))
STREAMING_DECISIONS = tuple(f'D{i}' for i in range(1, 8))
CLASSES = ('factual_information', 'general_advice', 'personal_advice', 'other')


def questions(config: dict) -> dict:
    """Return the resolved schema; checkpoints embed every instruction and criterion."""
    data = config['dataset']
    version = data.get('schema', LEGACY)
    if version in (D1_D6, *STREAMING_SCHEMAS):
        if not isinstance(data.get('questions'), dict):
            raise ValueError('D1–D6 requires resolved dataset.questions')
        return deepcopy(data['questions'])
    if version != LEGACY:
        raise ValueError(f'Unknown dataset schema: {version}')
    result = {'advice_type': {
        'type': 'choice', 'instructions': 'Classify only the AGENT response using CUSTOMER context.',
        'criteria': {label: data['descriptions'].get(label, label) for label in data['labels']},
    }}
    if data['multi_question']:
        for name, instruction in [
            ('product_recommendation', 'Does the agent recommend a financial product?'),
            ('uses_customer_circumstances', 'Does the agent use the customer’s circumstances?'),
        ]:
            result[name] = {'type': 'noul', 'instructions': instruction}
    return result


def primary_question(config: dict) -> str:
    version = config['dataset'].get('schema', LEGACY)
    return 'D7' if version in STREAMING_SCHEMAS else ('D6' if version == D1_D6 else 'advice_type')


def validate_questions(config: dict) -> None:
    qs = questions(config)
    primary = primary_question(config)
    for name, question in qs.items():
        if not isinstance(question, dict) or not isinstance(question.get('instructions'), str) or not question['instructions'].strip():
            raise ValueError(f'Question instructions required: {name}')
        if question.get('type') not in {'choice', 'noul'}:
            raise ValueError(f'Unsupported question type: {name}')
        if question['type'] == 'choice' and not isinstance(question.get('criteria'), dict):
            raise ValueError(f'Choice criteria required: {name}')
    if primary not in qs or qs[primary].get('type') != 'choice':
        raise ValueError('Primary classification question must be a choice')
    if set(qs[primary]['criteria']) != set(config['dataset']['labels']):
        raise ValueError('Question criteria differ from configured labels')
    if config['dataset'].get('schema', LEGACY) in (D1_D6, *STREAMING_SCHEMAS):
        order = STREAMING_DECISIONS if config['dataset']['schema'] in STREAMING_SCHEMAS else DECISIONS
        if config['dataset'].get('path'):
            raise ValueError('D1–D6 uses explicit dataset.files; clear DG_DATASET_PATH/dataset.path instead of silently ignoring it')
        if tuple(qs) != order or tuple(config['dataset']['labels']) != CLASSES:
            raise ValueError('D1–D6 requires six ordered decisions and the four canonical classes')
        if any(qs[d].get('type') != 'noul' for d in order[:-1]):
            raise ValueError('D1–D5 must be Boolean noul questions')


def contract(config: dict) -> dict:
    qs = questions(config)
    return {
        'version': config['dataset'].get('schema', LEGACY),
        'primary_question': primary_question(config), 'question_order': list(qs),
        'questions': qs, 'labels': list(config['dataset']['labels']),
    }


def contract_hash(config: dict) -> str:
    return fingerprint(contract(config))


def decision_violations(labels: dict) -> list[str]:
    """Diagnose contradictions; predictions are never rewritten to enforce labels."""
    errors = []
    if labels['D2'] and (not labels['D1'] or labels['D5']):
        errors.append('D2 requires D1=true and D5=false')
    if not labels['D2'] and (labels['D3'] or labels['D4']):
        errors.append('D3/D4 must be false when D2=false')
    implied = ('personal_advice' if labels['D3'] or labels['D4'] else 'general_advice') if labels['D2'] else ('factual_information' if labels['D5'] else 'other')
    final = 'D7' if 'D7' in labels else 'D6'
    if final == 'D7' and labels['D6'] and (not labels['D1'] or not labels['D2']):
        errors.append('Credit advice D6 requires D1=true and D2=true')
    if labels[final] != implied:
        errors.append(f'{final} disagrees with D2–D5')
    return errors
