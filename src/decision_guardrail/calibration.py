"""Calibration exclusively through Unsloth on the dedicated calibration split."""
from .dataset import build_dataset
from .utils import fingerprint


def calibrate(model, processor, splits: dict, config: dict | None = None) -> dict:
    """Fit native calibration and record its state for reload verification."""
    from .environment import set_offline
    set_offline()
    from unsloth import FastDecisionModel
    rows=splits['calibration']
    report=FastDecisionModel.calibrate(model,processor,build_dataset(rows,processor,model))
    state={k:v for k,v in model.decision_config.items() if 'temperature' in k or 'calibrat' in k}
    result = {'split':'calibration','records_sha256':fingerprint(rows),'count':len(rows),'report':report,'state':state,
              'question_ids':list(rows[0]['questions'])}
    if config is not None:
        from .schema import contract_hash
        result['schema_sha256'] = contract_hash(config)
    return result
