"""Per-class metrics, calibration error, timing and confidential-text-safe reports."""
import math
from pathlib import Path
import random
import time
from .utils import write_json, fingerprint
from .schema import questions, primary_question, contract_hash, decision_violations
from .inference import prediction_decisions


def metrics(expected: list[str], predictions: list[dict], labels: list[str], bins: int = 10) -> dict:
    """Compute one-vs-rest rates, macro scores, confusion matrix and top-label ECE."""
    if not expected or len(expected)!=len(predictions) or bins<1:
        raise ValueError('Nonempty aligned predictions and positive ECE bins required')
    matrix=[[0]*len(labels) for _ in labels]
    for y,p in zip(expected,predictions):
        matrix[labels.index(y)][labels.index(p['advice_type'])]+=1
    n=len(expected); per_class={}
    for i,label in enumerate(labels):
        tp=matrix[i][i]; fn=sum(matrix[i])-tp; fp=sum(row[i] for row in matrix)-tp; tn=n-tp-fp-fn
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        per_class[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'false_positive_rate':fp/(fp+tn) if fp+tn else None,'false_negative_rate':fn/(fn+tp) if fn+tp else None,'support':tp+fn}
    ece=0.
    for b in range(bins):
        values=[(p['probabilities'][p['advice_type']],int(y==p['advice_type'])) for y,p in zip(expected,predictions) if min(int(p['probabilities'][p['advice_type']]*bins),bins-1)==b]
        if values:
            ece+=len(values)/n*abs(sum(v[0] for v in values)/len(values)-sum(v[1] for v in values)/len(values))
    return {'count':n,'accuracy':sum(matrix[i][i] for i in range(len(labels)))/n,'macro':{k:sum(v[k] for v in per_class.values())/len(labels) for k in ['precision','recall','f1']},'per_class':per_class,'confusion_matrix':{'labels':labels,'rows_expected_columns_predicted':matrix},'expected_calibration_error':ece}


def accuracy_interval(expected: list[str], predictions: list[dict], samples: int, seed: int) -> list[float]:
    """Seeded nonparametric bootstrap accuracy interval; not a deployment guarantee."""
    rng=random.Random(seed); correct=[int(y==p['advice_type']) for y,p in zip(expected,predictions)]
    values=sorted(sum(rng.choices(correct,k=len(correct)))/len(correct) for _ in range(samples))
    return [values[int(.025*(samples-1))],values[int(.975*(samples-1))]]


def question_values(rows: list[dict], predictions: list[dict], name: str) -> tuple[list, list]:
    """Adapt native true/false option IDs for the same audited metric implementation."""
    def option(value):
        return str(value).lower() if type(value) is bool else value
    expected = [option(row['gold'][name]) for row in rows]
    answers = [prediction_decisions(p)[name] for p in predictions]
    projected = [{'advice_type': option(a['answer']), 'probabilities': a['probabilities']} for a in answers]
    return expected, projected


def decision_metrics(rows: list[dict], predictions: list[dict], config: dict) -> dict:
    """Report each decision, primary D6 metrics, and exact agreement of all six."""
    if not rows or len(rows) != len(predictions):
        raise ValueError('Nonempty aligned rows and predictions required')
    qs = questions(config)
    per_question = {}
    for name, spec in qs.items():
        expected, projected = question_values(rows, predictions, name)
        options = ['false', 'true'] if spec['type'] == 'noul' else list(spec['criteria'])
        per_question[name] = metrics(expected, projected, options, config['evaluation']['bins'])
        if config['evaluation']['bootstrap_samples']:
            per_question[name]['accuracy_bootstrap_95_interval'] = accuracy_interval(expected, projected, config['evaluation']['bootstrap_samples'], config['project']['seed'])
    primary = primary_question(config)
    report = dict(per_question[primary])
    correct = [sum(prediction_decisions(p)[q]['answer'] == row['gold'][q] for q in qs) for row,p in zip(rows,predictions)]
    report.update(
        primary_question=primary, schema_sha256=contract_hash(config),
        per_question=per_question, question_count=len(qs),
        decision_accuracy=sum(correct)/(len(rows)*len(qs)),
        exact_match_accuracy=sum(value == len(qs) for value in correct)/len(rows),
    )
    if 'D6' in qs:
        failures = sum(bool(decision_violations({q: a['answer'] for q,a in prediction_decisions(p).items()})) for p in predictions)
        report['logical_consistency'] = {'inconsistent_records': failures, 'consistent_fraction': 1-failures/len(rows)}
    return report


def evaluate(predictor, rows: list[dict], output: Path, cohort: str = 'test') -> dict:
    """Evaluate one held-out cohort and emit metrics plus error records."""
    config=predictor.config; predictions=[]; latencies=[]
    for suffix in ('.json', '_errors.jsonl', '.md'):
        if (output/f'{cohort}{suffix}').exists():
            raise FileExistsError('Evaluation report exists; choose a new report directory/cohort')
    from .environment import set_offline
    set_offline()
    import torch
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        torch.cuda.synchronize(); start=time.perf_counter()
        prediction=predictor.predict_state(row['state'])
        torch.cuda.synchronize(); latencies.append(time.perf_counter()-start); predictions.append(prediction)
    report=decision_metrics(rows,predictions,config)
    report.update(cohort=cohort, examples_per_second=len(rows)/sum(latencies),latency_seconds={'mean':sum(latencies)/len(latencies),'p95':sorted(latencies)[min(len(latencies)-1,math.ceil(.95*len(latencies))-1)]},gpu_peak_bytes=torch.cuda.max_memory_allocated(),dataset_sha256=fingerprint(rows))
    # Also exercise Unsloth's official evaluation API.
    from unsloth import FastDecisionModel
    from .dataset import build_dataset
    report['native']=FastDecisionModel.evaluate(predictor.model,predictor.processor,build_dataset(rows,predictor.processor,predictor.model))
    output.mkdir(parents=True,exist_ok=True); write_json(output/f'{cohort}.json',report)
    errors=[]
    for row,p in zip(rows,predictions):
        answers = prediction_decisions(p)
        failed = [q for q in row['gold'] if row['gold'][q] != answers[q]['answer']]
        if failed:
            error={'id':row.get('id'),'master_line':row.get('master_line'),'input_sha256':fingerprint(row['state']),
                   'incorrect_questions':failed,'expected':row['gold'],
                   'predicted':{q:a['answer'] for q,a in answers.items()},
                   'probabilities':{q:a['probabilities'] for q,a in answers.items()}}
            if config['evaluation']['include_text']:
                error['state']=row['state']
            errors.append(error)
    import json
    (output/f'{cohort}_errors.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in errors))
    lines=[f'# Evaluation: {cohort}', '', f'Examples: {len(rows)}; accuracy: {report["accuracy"]:.4f}; ECE: {report["expected_calibration_error"]:.4f}', '', '| Category | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|']
    lines += [f'| {k} | {v["precision"]:.4f} | {v["recall"]:.4f} | {v["f1"]:.4f} | {v["support"]} |' for k,v in report['per_class'].items()]
    lines += ['', f'Primary question: {report["primary_question"]}; all-decisions exact match: {report["exact_match_accuracy"]:.4f}', '',
              '| Decision | Accuracy | Macro F1 | ECE |', '|---|---:|---:|---:|']
    lines += [f'| {q} | {m["accuracy"]:.4f} | {m["macro"]["f1"]:.4f} | {m["expected_calibration_error"]:.4f} |' for q,m in report['per_question'].items()]
    lines += ['', 'Pilot and synthetic results do not establish production readiness.']
    (output/f'{cohort}.md').write_text('\n'.join(lines)+'\n')
    return report
