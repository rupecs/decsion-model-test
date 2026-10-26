"""Structured native decision predictions; enforcement is a separate operation."""
import math
from .dataset import questions, state_text, split_state
from .schema import D1_D6, D1_D7, decision_violations, contract_hash
from .environment import set_offline


def validate_answer(answer: dict, options: list[str], boolean: bool = False) -> dict:
    """Validate native probabilities without fabricating or renormalizing them."""
    if not isinstance(answer, dict) or 'answer' not in answer or not isinstance(answer.get('probabilities'), dict):
        raise ValueError('Missing native answer or probabilities')
    probs = answer['probabilities']
    value = answer['answer']
    if boolean and type(value) is not bool:
        raise ValueError('Native Boolean answer must be true/false')
    label = str(value).lower() if boolean else value
    if not isinstance(label, str) or set(probs)!=set(options) or label not in options:
        raise ValueError('Native prediction has unexpected labels')
    # Pinned Unsloth returns probabilities rounded to four decimal places.
    tolerance = max(1e-4, len(options)*0.00005) + 1e-9
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>tolerance:
        raise ValueError('Native prediction contains invalid probabilities')
    if probs[label] < max(probs.values())-1e-6:
        raise ValueError('Native answer differs from probability argmax')
    return {'answer': value, 'probabilities': dict(probs)}


def format_prediction(raw: dict, labels: list[str], version: str, question_schema: dict | None = None, schema_version: str | None = None) -> dict:
    """Expose all six native decisions; retain the explicit legacy schema API."""
    qs = question_schema or {'advice_type': {'type': 'choice', 'criteria': dict.fromkeys(labels)}}
    if not isinstance(raw, dict) or (question_schema is not None and set(raw) != set(qs)) or any(q not in raw for q in qs):
        raise ValueError('Native prediction questions differ from configured schema')
    answers = {q: validate_answer(raw[q], ['true','false'] if spec['type']=='noul' else list(spec['criteria']), spec['type']=='noul') for q,spec in qs.items()}
    if 'D6' in qs:
        final = 'D7' if 'D7' in qs else 'D6'
        values = {q: answer['answer'] for q,answer in answers.items()}
        violations = decision_violations(values)
        return {
            'schema_version': schema_version or (D1_D7 if final == 'D7' else D1_D6), 'labels': values, 'decisions': answers,
            'classification': values[final], 'probabilities': answers[final]['probabilities'],
            'model_version': version,
            'consistency': {'consistent': not violations, 'violations': violations},
        }
    result={'advice_type':answers['advice_type']['answer'],'probabilities':answers['advice_type']['probabilities'],'model_version':version}
    if len(raw)>1:
        result['additional_questions']={k:v for k,v in (answers if question_schema else raw).items() if k!='advice_type'}
    return result


def prediction_decisions(prediction: dict) -> dict:
    if 'decisions' in prediction:
        return prediction['decisions']
    return {'advice_type': {'answer': prediction['advice_type'], 'probabilities': prediction['probabilities']}, **prediction.get('additional_questions', {})}


def compare_predictions(actual: dict, expected: dict, tolerance: float = 1e-4) -> float:
    """Reload verification includes every Boolean and choice probability."""
    if expected.get('schema_sha256') is not None and actual.get('schema_sha256') != expected['schema_sha256']:
        raise RuntimeError('Reload changed the decision schema')
    observed, saved = prediction_decisions(actual), prediction_decisions(expected)
    if set(observed) != set(saved):
        raise RuntimeError('Reload changed the predicted questions')
    delta = 0.0
    for name, target in saved.items():
        value = observed[name]
        if type(value['answer']) is not type(target['answer']) or value['answer'] != target['answer']:
            raise RuntimeError(f'Reload changed predicted label: {name}')
        if set(value['probabilities']) != set(target['probabilities']):
            raise RuntimeError(f'Reload changed probability options: {name}')
        differences = [abs(value['probabilities'][key]-prob) for key,prob in target['probabilities'].items()]
        if any(not math.isfinite(difference) for difference in differences):
            raise RuntimeError(f'Reload contains non-finite probabilities: {name}')
        delta = max(delta, max(differences))
    if delta > tolerance + 1e-9:
        raise RuntimeError(f'Reload probability drift {delta} exceeds {tolerance}')
    return delta


class Predictor:
    """Load once, reuse processor and model across predictions."""
    def __init__(self, config: dict, checkpoint: str | None = None, model=None, processor=None):
        set_offline()
        self.config=config
        if model is None:
            if not checkpoint:
                raise ValueError('Inference requires a trained checkpoint')
            from .model_loader import load_model
            from .checkpoint import verify_checkpoint
            verify_checkpoint(checkpoint,config)
            model,processor=load_model(config,checkpoint)
        self.model,self.processor=model,processor
        from unsloth import FastDecisionModel
        FastDecisionModel.for_inference(model)

    def predict(self, customer_context: str, agent_response: str) -> dict:
        """Return native calibrated probabilities when saved calibration is present."""
        return self.predict_state(state_text(customer_context,agent_response))

    def predict_state(self, state: str) -> dict:
        """Classify the exact stored conversation state without reparsing earlier turns."""
        split_state(state)
        from unsloth import FastDecisionModel
        from .dataset import build_dataset
        qs=questions(self.config)
        # Validate length through the same native encoder used for training.
        dummy={'state':state,'questions':qs,'gold':{k:(next(iter(q['criteria'])) if q['type']=='choice' else False) for k,q in qs.items()}}
        if 'D6' in qs:
            dummy['gold']['D7' if 'D7' in qs else 'D6'] = 'other'
        build_dataset([dummy],self.processor,self.model)
        raw=FastDecisionModel.predict(self.model,self.processor,state,qs)
        result = format_prediction(raw,self.config['dataset']['labels'],self.config['project']['model_version'],qs,self.config['dataset'].get('schema'))
        result['schema_sha256'] = contract_hash(self.config)
        return result


def predict(customer_context: str, agent_response: str, config: dict) -> dict:
    """Convenience single-shot API; config paths.output_dir locates the checkpoint."""
    return Predictor(config,config['paths']['output_dir']).predict(customer_context,agent_response)
