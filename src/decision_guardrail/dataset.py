"""Strict local financial-content records, grouped splits, official tokenization."""
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from .utils import fingerprint
from .schema import DECISIONS, STREAMING_DECISIONS, CLASSES, D1_D6, D1_D7, STREAMING_SCHEMAS, questions, decision_violations

SPLITS = ('train','validation','calibration','test')


def state_text(customer_context: str, agent_response: str) -> str:
    """Retain explicit role boundaries in the classification input."""
    if not isinstance(customer_context, str) or not isinstance(agent_response, str) or not customer_context.strip() or not agent_response.strip():
        raise ValueError('Nonempty customer context and agent response required')
    # Canonicalise only unindented speaker boundaries. Quoted/indented role
    # strings remain content. ASSISTANT is the UI-facing alias for AGENT.
    # Ordinary canonical v4 inputs retain their exact bytes.
    context = customer_context.replace('\r\n', '\n')
    context = re.sub(r'(?im)^(customer|agent|assistant):(?=\s|$)',
                     lambda match: 'CUSTOMER:' if match[1].lower() == 'customer' else 'AGENT:', context)
    if context.startswith('AGENT:'):
        raise ValueError('Conversation context must begin with the customer')
    if re.search(r'(?im)^(customer|agent|assistant):', agent_response):
        raise ValueError('Target contains an ambiguous speaker boundary; quote or indent literal role strings')
    if not context.startswith('CUSTOMER: '):
        context = 'CUSTOMER: ' + context
    return context + '\nAGENT: ' + agent_response


def split_state(state: str) -> tuple[str, str]:
    """The final AGENT boundary is the target; earlier agent turns stay in context."""
    if not isinstance(state, str):
        raise ValueError('State must be text')
    context, boundary, response = state.rpartition('\nAGENT: ')
    if not boundary or not context.startswith('CUSTOMER: ') or not context.removeprefix('CUSTOMER: ').strip() or not response.strip():
        raise ValueError('Missing nonempty CUSTOMER context or final AGENT response')
    return context, response


def validate_record(record: dict, labels: list[str]) -> None:
    """Validate labels and context; exact sequence fitting happens with Unsloth."""
    if not isinstance(record,dict) or not isinstance(record.get('state'),str):
        raise ValueError('Each record requires string state')
    split_state(record['state'])
    qs, gold=record.get('questions'),record.get('gold')
    if not isinstance(qs,dict) or not qs or not isinstance(gold,dict) or set(qs)!=set(gold):
        raise ValueError('questions and gold must have matching nonempty keys')
    primary = 'D7' if 'D7' in qs else ('D6' if 'D6' in qs else 'advice_type')
    order = STREAMING_DECISIONS if primary == 'D7' else DECISIONS
    if primary not in qs:
        raise ValueError('advice_type or D1–D6 questions required')
    if primary in ('D6', 'D7') and (tuple(qs) != order or set(labels) != set(CLASSES)):
        raise ValueError('Six ordered D1–D6 questions and canonical classes required')
    for name,q in qs.items():
        if not isinstance(q,dict) or not isinstance(q.get('instructions'),str) or not q['instructions'].strip():
            raise ValueError('Question instructions required')
        if q.get('type')=='choice':
            criteria=q.get('criteria',{})
            if not isinstance(criteria,dict) or len(criteria)<2 or gold[name] not in criteria:
                raise ValueError('Invalid choice criteria or gold label')
            if name==primary and (set(criteria)!=set(labels) or gold[name] not in labels):
                raise ValueError('Unexpected advice labels')
        elif q.get('type')=='noul':
            if type(gold[name]) is not bool:
                raise ValueError('noul gold must be Boolean')
        else:
            raise ValueError('Supported question types: choice and noul')
    if qs[primary]['type']!='choice':
        raise ValueError('Primary classification must be multiclass choice')
    if primary in ('D6', 'D7'):
        if any(qs[d]['type'] != 'noul' for d in order[:-1]):
            raise ValueError('D1–D5 must use Boolean noul questions')
        violations = decision_violations(gold)
        if violations:
            raise ValueError('Contradictory D1–D6 gold labels: ' + '; '.join(violations))


def record_key(record: dict) -> str:
    """Detect identical inputs independent of labels or record IDs."""
    return fingerprint(' '.join(record['state'].casefold().split()))


def load_records(path: Path, labels: list[str]) -> list[dict]:
    """Read JSONL or a channel directory containing exactly one JSONL file."""
    if path.is_dir():
        files=sorted(path.glob('*.jsonl'))
        if len(files)!=1:
            raise ValueError('Input channel must contain exactly one JSONL file')
        path=files[0]
    result=[]; seen=set()
    for line_no,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            continue
        try:
            row=json.loads(line); validate_record(row,labels)
            key=record_key(row)
            if key in seen:
                raise ValueError('Duplicate input')
            seen.add(key); result.append(row)
        except (ValueError,TypeError) as exc:
            raise ValueError(f'{path.name}:{line_no}: {exc}') from exc
    if not result:
        raise ValueError('Empty dataset')
    return result


def grouped_split(records: list[dict], sizes: dict, seed: int) -> dict[str,list[dict]]:
    """Keep conversation/paraphrase group IDs together; never cut a group.

    Exact counts are required. If groups cannot fill them, provide preassigned
    split files instead of silently changing the intended experimental design.
    """
    n=len(records)
    if all(isinstance(v,int) for v in sizes.values()):
        counts=dict(sizes)
        if sum(counts.values())!=n:
            raise ValueError(f'Split counts total {sum(counts.values())}, dataset contains {n}')
    else:
        counts={s:int(n*sizes[s]) for s in SPLITS}
        for s in sorted(SPLITS,key=lambda s:n*sizes[s]-counts[s],reverse=True)[:n-sum(counts.values())]:
            counts[s]+=1
    if any(v<1 for v in counts.values()):
        raise ValueError('Every split must be nonempty')
    groups=defaultdict(list)
    parents={}
    def find(key):
        parents.setdefault(key,key)
        if parents[key]!=key:
            parents[key]=find(parents[key])
        return parents[key]
    identifiers=[]
    for row in records:
        ids=[f'{field}:{row[field]}' for field in ['group_id','conversation_id'] if row.get(field)]
        if not ids:
            raise ValueError('group_id or conversation_id required to prevent leakage')
        for key in ids[1:]:
            parents[find(key)]=find(ids[0])
        identifiers.append(ids[0])
    for key,row in zip(identifiers,records):
        groups[find(key)].append(row)
    rng=random.Random(seed); values=list(groups.values()); rng.shuffle(values)
    values.sort(key=len,reverse=True)
    result={s:[] for s in SPLITS}
    remaining=dict(counts)
    for group in values:
        candidates=[s for s in SPLITS if remaining[s]>=len(group)]
        if not candidates:
            raise ValueError('Groups cannot fill exact split counts; provide explicit split files')
        chosen=max(candidates,key=lambda s:remaining[s])
        result[chosen].extend(group); remaining[chosen]-=len(group)
    return result


def assert_disjoint(splits: dict) -> None:
    """Reject shared content and conversation/paraphrase groups across splits."""
    seen_keys=set(); seen_groups=set()
    for rows in splits.values():
        keys={record_key(r) for r in rows}
        groups={f'{field}:{r[field]}' for r in rows for field in ['group_id','conversation_id'] if r.get(field)}
        groups |= {group for row in rows for group in row.get('leakage_groups', [])}
        if any(not r.get('group_id') and not r.get('conversation_id') for r in rows):
            raise ValueError('Every record requires a grouping identifier')
        if len(keys)!=len(rows) or keys & seen_keys or groups & seen_groups:
            raise ValueError('Dataset leakage: duplicate input or shared conversation group')
        seen_keys |= keys; seen_groups |= groups


def load_splits(config: dict) -> dict:
    """Resolve four explicit files or reproducibly split one approved source."""
    d=config['dataset']; files=d['files']
    if d.get('schema') in STREAMING_SCHEMAS:
        from .streaming_data import load_streaming_splits
        result = load_streaming_splits(config)
    elif d.get('schema') == D1_D6:
        from .pilot_data import load_decision_splits
        result = load_decision_splits(config)
    elif files:
        if set(files)!=set(SPLITS):
            raise ValueError('Supply all four split files/channels, including calibration')
        result={s:load_records(resolve_data(files[s],config),d['labels']) for s in SPLITS}
    else:
        rows=load_records(resolve_data(d['path'],config),d['labels'])
        result=grouped_split(rows,d['splits'],config['project']['seed'])
    assert_disjoint(result)
    total=sum(map(len,result.values()))
    if d.get('expected_total') and total!=d['expected_total']:
        raise ValueError(f'Expected {d["expected_total"]} examples, got {total}; supply the intended dataset')
    if all(type(v) is int for v in d['splits'].values()):
        if {s:len(rows) for s,rows in result.items()} != d['splits']:
            raise ValueError('Loaded split counts differ from configured split counts')
    expected=questions(config)
    for rows in result.values():
        for row in rows:
            if row['questions']!=expected or list(row['questions']) != list(expected):
                raise ValueError('Dataset question schema differs from active configured schema')
    return result


def resolve_data(source: str, config: dict) -> Path:
    """Stage an approved S3 JSONL object through prefix staging when requested."""
    if source.startswith('s3://'):
        from .storage import stage_s3_file
        return stage_s3_file(source,Path(config['model']['cache_dir'])/'datasets',region=config['storage']['region'])
    return Path(source)


def build_dataset(rows: list[dict], processor, model, coverage: dict | None = None) -> list:
    """Use official encoding and reject truncation/skips rather than losing labels."""
    from .environment import set_offline
    set_offline()
    from unsloth import FastDecisionModel
    from unsloth.models.clef import encode_record
    tokenizer=getattr(processor,'tokenizer',processor)
    max_len=model.decision_config['max_len']
    # Compare official encodings against a larger cap; includes question/schema tokens.
    native_rows = [{key: row[key] for key in ('state', 'questions', 'gold')} for row in rows]
    lengths = []
    for row in native_rows:
        full=encode_record(tokenizer,row,max_length=max_len+1,processor=processor if hasattr(processor,'tokenizer') else None)
        if len(full.input_ids)>max_len:
            raise ValueError('Record exceeds max_seq_length including schema; shorten or explicitly curate it')
        lengths.append(len(full.input_ids))
    items,report=FastDecisionModel.build_dataset(native_rows,processor,model)
    if report.get('skipped') or report.get('truncated') or len(items)!=len(rows):
        raise ValueError(f'Unsloth dataset conversion rejected records: {report}')
    if coverage is not None:
        ordered = sorted(lengths)
        coverage.update(records=len(rows), native_max_seq_length=max_len,
            minimum_tokens=min(ordered), maximum_tokens=max(ordered),
            median_tokens=ordered[len(ordered)//2], p95_tokens=ordered[min(len(ordered)-1,int(.95*len(ordered)))],
            skipped=0, truncated=0, method='official native encoder including all questions')
    return items


def split_manifest(splits: dict) -> dict:
    """Record split assignments and label definitions without customer text."""
    return {s:[{'id':r.get('id'), 'group':r.get('group_id',r.get('conversation_id')), 'master_line':r.get('master_line'), 'leakage_groups':r.get('leakage_groups', []), 'input_sha256':record_key(r), 'record_sha256':fingerprint(r)} for r in rows] for s,rows in splits.items()}
