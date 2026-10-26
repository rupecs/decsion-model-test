"""Versioned construction release: source replay, explicit corrections and context contrasts."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import yaml

from ..dataset import state_text, split_state, validate_record
from ..schema import CLASSES, decision_violations
from .context_v5_review import sha, digest, write_json, write_rows
from .context_v5_content import make_cases
from .streaming_15k_catalog import TOPICS
from .streaming_15k_v4 import normalise, lexical_baseline
from .streaming_15k_v4_content import render, context as original_context

VERSION='banking_streaming_d1_d7_v5_context_draft'
GENERATOR='streaming-context-local-compositional-v5'
SPLITS=('train','validation','calibration','test')
STATUS='user_authorized_construction_release_pending_independent_review'


def native_questions(root):
    schema=yaml.safe_load((root/'config/dataset/question_schema_streaming_v5.yaml').read_text())
    result={d:dict(type='noul',instructions=q['instructions']) for d,q in schema['decisions'].items() if d!='D7'}
    result['D7']=dict(type='choice',instructions=schema['decisions']['D7']['instructions'],criteria={
        'factual_information':'Substantive facts or procedural help with no endorsed product advice in this target.',
        'general_advice':'Product advice without demonstrated actual-basis use or individual suitability in conversation context.',
        'personal_advice':'Product advice using actual circumstances or asserting/implying individual suitability in conversation context.',
        'other':'No substantive facts, procedural help or endorsed product advice expressed so far.'})
    return schema,result


def annotation(profile, rationale, target, context):
    return dict(profile=profile,visible_evidence=target,rationale=rationale,
        rule_version=VERSION,status='construction_audited_pending_independent_annotation',
        reviewer='Codex construction audit',independent_review=False,
        evidence=dict(target_spans=[dict(start=0,end=len(target),text=target)],
            context_turns=[dict(id=f'context-{i}',text=line) for i,line in enumerate(context.splitlines(),1)]))


def source_rows(root):
    source=root/'data/generated/streaming_15000_v4/records.jsonl'
    originals=[json.loads(line) for line in source.read_text().splitlines()]
    corrections={r['source_id']:r for r in json.loads((root/'data/authored/streaming_context_v5_release/source_label_corrections.json').read_text())}
    topics={t.key:t for t in TOPICS};cache={};raw=[];dispositions=[]
    if not set(corrections)<=set(r['id'] for r in originals):raise ValueError('Unknown correction source')
    for line,r in enumerate(originals,1):
        meta=r['metadata'];ref=meta['source_refs'][0]
        case=int(meta['customer_situation_id'].rsplit(':',1)[1]);topic=topics[meta['contrast_family']]
        key=(topic.key,case,meta['construction_recipe'],meta['split'])
        if key not in cache:
            cache[key]=(original_context(topic,case)[0],render(topic,case,meta['construction_recipe'],meta['split']))
        ctx,points=cache[key];point=points[ref['checkpoint_index']]
        # Provenance replay is a structural check, not independent semantic review.
        if ctx!=r['input']['customer_context'] or point['text']!=r['input']['agent_response'] or point['profile']!=r['annotation']['profile']:
            raise ValueError('Frozen source replay differs: '+r['id'])
        change=corrections.get(r['id'])
        labels=deepcopy(r['labels']);reason=r['annotation']['rationale']
        if change:
            if change['expected_input']!=r['input'] or change['before_labels']!=labels:
                raise ValueError('Correction fingerprint precondition differs')
            labels=deepcopy(change['after_labels']);reason=change['rationale']
        unit='v4:'+ref['scenario_id']
        raw.append(dict(id='sctxv5-'+digest(r['input'])[:24],schema_version=VERSION,input=deepcopy(r['input']),
            labels=labels,advice_present=labels['D2'],annotation=annotation(r['annotation']['profile'],reason,r['input']['agent_response'],ctx),
            metadata=dict(parent_topic=topic.key,original_split=meta['split'],origin='retained_v4',
                units=[unit],source_ids=[r['id']],source_refs=deepcopy(meta['source_refs']),
                scenario_product_domains=meta['scenario_product_domains'],
                customer_basis_type=meta['customer_basis_type'],customer_disclosure=meta['customer_disclosure'],
                construction_recipe=meta['construction_recipe'],coverage_boundary=meta['coverage_boundary'],
                recommendation_withdrawn_in_target=meta['recommendation_withdrawn_in_target'],
                recommendation_withdrawn=meta['recommendation_withdrawn'],
                lineage=[dict(version=r['schema_version'],source_id=r['id'],master_line=line,
                    original_split=meta['split'],input_sha256=digest(r['input']),record_sha256=digest(r))],
                synthetic=True,independent_annotation=False)))
        dispositions.append(dict(source_version=r['schema_version'],source_id=r['id'],source_master_line=line,
            source_split=meta['split'],source_family=meta['split_group'],source_record_sha256=digest(r),
            before_labels=r['labels'],after_labels=labels,disposition='correct' if change else 'keep',
            rationale=reason,review_method='Authored source/profile replay and construction-rule audit; explicit source-ID correction ledger.',
            status='construction_audit_not_independent_annotation',reviewer='Codex construction audit',
            evidence=dict(context=ctx,target=r['input']['agent_response']),independent_review=False))
    return raw,dispositions


def labels_for(profile,credit):
    bits={'general':(1,1,0,0,0,credit),'implicit':(1,1,0,1,0,credit),
          'linked':(1,1,1,1,0,credit),'fact':(1,0,0,0,1,0),
          'quote':(1,0,0,0,0,0),'mention':(1,0,0,0,0,0)}[profile]
    labels={f'D{i}':bool(v) for i,v in enumerate(bits,1)}
    labels['D7']=('personal_advice' if labels['D3'] or labels['D4'] else 'general_advice') if labels['D2'] else ('factual_information' if labels['D5'] else 'other')
    return labels


def addition_rows(root):
    raw=[];families=make_cases(root)
    for family in families:
        unit='addition:'+family['id'];points=family['points']
        for i,p in enumerate(points):
            ctx,_=split_state(state_text(p['context'],p['text']))
            labels=deepcopy(p['labels']) if 'labels' in p else labels_for(p['profile'],family['credit'])
            trajectory=unit if family['prefix'] else unit+':target-'+str(i)
            checkpoint=i if family['prefix'] else 0;total=len(points) if family['prefix'] else 1
            inp=dict(customer_context=ctx,agent_response=p['text'])
            raw.append(dict(id='sctxv5-'+digest(inp)[:24],schema_version=VERSION,input=inp,labels=labels,
                advice_present=labels['D2'],annotation=annotation(p['profile'],p['rationale'],p['text'],ctx),
                metadata=dict(parent_topic=family['parent'],origin=family['origin'],units=[unit],source_ids=[],
                    source_refs=[dict(scenario_id=trajectory,checkpoint_index=checkpoint,is_final=checkpoint==total-1,total_checkpoints=total,
                        value_added=family['design'])],scenario_product_domains=[family['domain']],
                    customer_basis_type='authored_context_contrast',customer_disclosure=None,
                    construction_recipe=family['purpose'],coverage_boundary=family['purpose'],
                    recommendation_withdrawn_in_target=family['prefix'] and p['text'].endswith('I withdraw that recommendation.'),
                    recommendation_withdrawn=family['prefix'] and p['text'].endswith('I withdraw that recommendation.'),
                    lineage=[dict(version='context_v5_addition',family_id=family['id'],point=i,
                        review_source_id=p.get('review_source_id'),review_fingerprint=p.get('review_fingerprint'))],
                    synthetic=True,independent_annotation=False)))
    return raw,families


class Groups:
    def __init__(self):self.parents={}
    def find(self,x):
        self.parents.setdefault(x,x)
        if self.parents[x]!=x:self.parents[x]=self.find(self.parents[x])
        return self.parents[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parents[max(a,b)]=min(a,b)


def merge_rows(raw):
    accepted={};conflicts=set();duplicates=0
    for r in raw:
        key=digest(r['input'])
        if key not in accepted:accepted[key]=deepcopy(r);continue
        existing=accepted[key]
        if existing['labels']!=r['labels']:
            conflicts.update(existing['metadata']['units']+r['metadata']['units']);continue
        duplicates+=1
        for field in ('units','source_ids','source_refs','lineage'):
            for item in r['metadata'][field]:
                if item not in existing['metadata'][field]:existing['metadata'][field].append(item)
        existing['metadata']['recommendation_withdrawn_in_target'] |= r['metadata']['recommendation_withdrawn_in_target']
    if conflicts:raise ValueError('Conflicting construction labels require explicit adjudication: '+str(sorted(conflicts)))
    return list(accepted.values()),duplicates


def assign_splits(rows):
    groups=Groups();target_owner={};context_owner={}
    for r in rows:
        parent=r['metadata']['parent_topic'];groups.find(parent)
        key=normalise(r['input']['agent_response'])
        # Group exact complete/substantive targets. Non-substantive unfinished
        # fragments are reported separately; they cannot join every topic into one.
        complete=r['labels']['D2'] or r['labels']['D5'] or any(ref['is_final'] for ref in r['metadata']['source_refs'])
        if complete:
            if key in target_owner:groups.union(parent,target_owner[key])
            else:target_owner[key]=parent
        context_key=normalise(r['input']['customer_context'])
        if context_key in context_owner:groups.union(parent,context_owner[context_key])
        else:context_owner[context_key]=parent
    votes=defaultdict(Counter)
    for r in rows:
        if r['metadata'].get('original_split'):votes[groups.find(r['metadata']['parent_topic'])][r['metadata']['original_split']]+=1
    owners={g:max(SPLITS,key=lambda s:counts[s]) for g,counts in votes.items()}
    for r in rows:
        group=groups.find(r['metadata']['parent_topic'])
        split=owners.get(group,'train');meta=r['metadata']
        meta.update(split=split,split_group=group,contrast_family=group,
            customer_situation_id='ctx-'+digest({'context':r['input']['customer_context']})[:20])
    if set(r['metadata']['split'] for r in rows)!=set(SPLITS):
        raise ValueError('Family linkage prevents four nonempty partitions; cannot pretend exact quotas')
    return dict(component_owners=owners,components=len(owners))


def lexical_candidates(rows):
    final=[r for r in rows if any(ref['is_final'] for ref in r['metadata']['source_refs'])]
    index=defaultdict(set);grams=[];pairs=[]
    for i,r in enumerate(final):
        words=normalise(r['input']['agent_response']).split()
        gs={tuple(words[k:k+5]) for k in range(max(0,len(words)-4))}
        overlaps=Counter(j for gram in gs for j in index[gram])
        for j,n in overlaps.items():
            other=final[j]
            if other['metadata']['split']==r['metadata']['split']:continue
            union=len(gs)+len(grams[j])-n
            if union and n/union>=.65:pairs.append((other,r,n/union))
        for gram in gs:index[gram].add(i)
        grams.append(gs)
    return pairs


def resolve_lexical(rows):
    excluded=[];initial=0
    for iteration in range(5):
        pairs=lexical_candidates(rows)
        if not pairs:return rows,excluded,initial
        initial+=len(pairs);units=set()
        for a,b,score in pairs:
            def priority(r):
                m=r['metadata']
                return (not bool(m['source_ids']),m.get('original_split',m['split'])!=m['split'],r['id'])
            chosen=max((a,b),key=priority)
            units.update(chosen['metadata']['units'])
        # A shared deduplicated row belongs to every parent unit; quarantine its
        # complete related units, not just a conveniently selected checkpoint.
        while True:
            linked={u for r in rows if units.intersection(r['metadata']['units']) for u in r['metadata']['units']}
            if linked<=units:break
            units|=linked
        keep=[]
        for r in rows:
            if units.intersection(r['metadata']['units']):
                excluded.append(dict(record=r,reason='Cross-partition completed-target word-5gram Jaccard >=0.65; complete linked construction units quarantined.',iteration=iteration))
            else:keep.append(r)
        rows=keep
    raise ValueError('Lexical candidates remain unresolved')


def trajectories_for(rows):
    result=defaultdict(list)
    for r in rows:
        for ref in r['metadata']['source_refs']:result[ref['scenario_id']].append((ref['checkpoint_index'],r))
    trajectories=[]
    for scenario,points in sorted(result.items()):
        points.sort(key=lambda p:p[0]);seq=[r for _,r in points]
        if len({r['input']['customer_context'] for r in seq})!=1:raise ValueError('History changed inside accumulated target')
        for i,r in enumerate(seq):
            if i and (not r['input']['agent_response'].startswith(seq[i-1]['input']['agent_response']) or r['input']==seq[i-1]['input']):
                raise ValueError('Non-prefix target trajectory')
            if i and seq[i-1]['labels']['D2'] and not r['labels']['D2']:raise ValueError('Advice exposure erased')
            for ref in r['metadata']['source_refs']:
                if ref['scenario_id']==scenario and (ref['checkpoint_index']!=i or ref['total_checkpoints']!=len(seq) or ref['is_final']!=(i==len(seq)-1)):
                    raise ValueError('Trajectory lineage mismatch')
        trajectories.append(dict(id=scenario,family=seq[0]['metadata']['split_group'],split=seq[0]['metadata']['split'],snapshot_ids=[r['id'] for r in seq]))
    return trajectories


def quality(rows,trajectories):
    seen=set();group_splits=defaultdict(set);target_splits=defaultdict(set);partial_splits=defaultdict(set)
    qs_dummy={d:dict(type='noul',instructions=d) for d in ('D1','D2','D3','D4','D5','D6')}
    qs_dummy['D7']=dict(type='choice',instructions='Classify',criteria=dict.fromkeys(CLASSES))
    for r in rows:
        if r['schema_version']!=VERSION or decision_violations(r['labels']):raise ValueError('Schema/label inconsistency')
        inp=digest(r['input'])
        if inp in seen:raise ValueError('Duplicate full input')
        seen.add(inp);m=r['metadata'];group_splits[m['split_group']].add(m['split'])
        validate_record(dict(state=state_text(**r['input']),questions=qs_dummy,gold=r['labels']),list(CLASSES))
        key=normalise(r['input']['agent_response'])
        complete=r['labels']['D2'] or r['labels']['D5'] or any(x['is_final'] for x in m['source_refs'])
        (target_splits if complete else partial_splits)[key].add(m['split'])
    if any(len(s)>1 for s in group_splits.values()) or any(len(s)>1 for s in target_splits.values()):raise ValueError('Exact complete-target or family leakage')
    by_id={r['id']:r for r in rows};seqs=[[by_id[i] for i in t['snapshot_ids']] for t in trajectories]
    advice=[s for s in seqs if s[-1]['labels']['D2']]
    for split in SPLITS:
        selected=[r for r in rows if r['metadata']['split']==split]
        if set(r['labels']['D7'] for r in selected)!=set(CLASSES):raise ValueError('A partition lacks a class')
        if not any(r['labels']['D3'] for r in selected) or not any(r['labels']['D6'] for r in selected):raise ValueError('A partition lacks D3 or credit advice')
    return dict(records=len(rows),scenarios=len(trajectories),
        split_counts={s:sum(r['metadata']['split']==s for r in rows) for s in SPLITS},
        class_counts=dict(Counter(r['labels']['D7'] for r in rows)),
        class_counts_by_split={s:dict(Counter(r['labels']['D7'] for r in rows if r['metadata']['split']==s)) for s in SPLITS},
        binary_counts={d:dict(Counter(str(r['labels'][d]).lower() for r in rows)) for d in [f'D{i}' for i in range(1,7)]},
        credit_advice_rows=sum(r['labels']['D6'] for r in rows),
        D3_rows=sum(r['labels']['D3'] for r in rows),
        D4_only_personal_rows=sum(r['labels']['D4'] and not r['labels']['D3'] for r in rows),
        checkpoint_count_distribution=dict(Counter(len(s) for s in seqs)),
        final_checkpoint_first_advice_fraction=sum(s[-1]['labels']['D2'] and not any(r['labels']['D2'] for r in s[:-1]) for s in advice)/len(advice),
        no_advice_trajectories=sum(not any(r['labels']['D2'] for r in s) for s in seqs),
        independently_annotated_records=0,exact_complete_targets_crossing_splits=0,
        shared_non_substantive_unfinished_targets=sum(len(s)>1 for s in partial_splits.values()),
        semantic_embedding_screen=False,native_token_coverage='pending actual native encoder',
        independent_challenge_status='not authored or evaluated by this release',
        split_status='Development partitions; exposed legacy/review material. Not an independent challenge.',
        limitations=['Construction audit and user permission are not independent annotation or legal approval.',
          'Shared product catalogue and compositional designs remain; no semantic independence claim.',
          'Short non-substantive unfinished target strings can cross partitions; full input, complete targets and linked families do not.',
          'No native token validation, model evaluation, training or GPU work was performed.'])


def build(root,output):
    if output.exists():raise ValueError('Use a new release path; never overwrite artifacts')
    original,dispositions=source_rows(root);additions,families=addition_rows(root)
    rows,deduplicated=merge_rows(original+additions)
    assignment=assign_splits(rows)
    rows,quarantine,candidates=resolve_lexical(rows)
    # All source snapshots receive an explicit construction-audit disposition.
    retained_source={sid:r for r in rows for sid in r['metadata']['source_ids']}
    for d in dispositions:
        if d['source_id'] not in retained_source:
            d.update(disposition='quarantine',reason='Linked unit excluded by cross-partition lexical candidate resolution.',after_labels=None)
        else:
            r=retained_source[d['source_id']];d.update(release_id=r['id'],release_split=r['metadata']['split'])
    trajectories=trajectories_for(rows)
    report=quality(rows,trajectories);schema,qs=native_questions(root)
    report.update(generator=GENERATOR,source_dispositions=dict(Counter(d['disposition'] for d in dispositions)),
        source_records=15000,retained_source_records=len(retained_source),
        candidate_addition_snapshots=len(additions),candidate_addition_families=len(families),
        accepted_addition_units=len({u for r in rows for u in r['metadata']['units'] if u.startswith('addition:')}),
        new_unique_records=sum(not r['metadata']['source_ids'] for r in rows),
        addition_primary_purposes=dict(Counter(f['purpose'] for f in families)),
        merged_duplicate_inputs=deduplicated,quarantined_unique_records=len(quarantine),
        cross_partition_lexical_candidates_resolved=candidates,cross_partition_lexical_near_duplicate_count=0,
        family_assignment=assignment,
        lexical_baselines={field:lexical_baseline(rows,field) for field in ('customer_context','agent_response')})
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='context-v5-',dir=output.parent) as tmp:
        stage=Path(tmp)/'release';stage.mkdir()
        write_rows(stage/'records.jsonl',rows);write_json(stage/'trajectories.json',trajectories)
        write_json(stage/'quality_report.json',report);write_json(stage/'questions.json',schema);write_json(stage/'native_questions.json',qs)
        line_index={r['id']:i for i,r in enumerate(rows,1)};write_json(stage/'line_index.json',line_index)
        for split in SPLITS:
            selected=[r for r in rows if r['metadata']['split']==split]
            for folder in ('splits','clef','clef_index'):(stage/folder).mkdir(exist_ok=True)
            write_rows(stage/'splits'/f'{split}.jsonl',selected)
            write_rows(stage/'clef'/f'{split}.jsonl',[dict(state=state_text(**r['input']),questions=qs,gold=r['labels']) for r in selected])
            write_rows(stage/'clef_index'/f'{split}.jsonl',[dict(export_line=i,record_id=r['id'],records_jsonl_line=line_index[r['id']],recommendation_withdrawn_in_target=r['metadata']['recommendation_withdrawn_in_target']) for i,r in enumerate(selected,1)])
        write_rows(stage/'source_dispositions.jsonl',dispositions)
        write_rows(stage/'label_changes.jsonl',[d for d in dispositions if d['before_labels']!=d.get('after_labels') and d['after_labels'] is not None])
        write_rows(stage/'quarantine.jsonl',quarantine)
        write_json(stage/'addition_families.json',[dict(id=f['id'],parent=f['parent'],purpose=f['purpose'],design=f['design'],prefix=f['prefix'],origin=f['origin']) for f in families])
        write_rows(stage/'review_queue.jsonl',[dict(record_id=r['id'],records_jsonl_line=line_index[r['id']],status='pending_independent_annotation',independent_review=False) for r in rows])
        (stage/'annotation_standard.md').write_text((root/'docs/annotation_standard_streaming_v5.md').read_text())
        (stage/'README.md').write_text('# Streaming context v5 construction release\n\n'+
            f'{len(rows)} unique snapshots. Corrected/retained v4 plus accepted context additions.\n\n'+
            'All four native Clef exports use the v5 question contract. No model weights or runtime libraries are bundled.\n'+
            'See quality_report.json for actual counts, source dispositions, exclusions, leakage checks and lexical baselines.\n'+
            'These are user-authorised construction labels pending independent annotation. Development splits are not a fresh challenge.\n'+
            'Actual native token coverage and a new contract-specific GPU smoke remain required before training. No GPU was started.\n')
        payload=dict(rows=rows,questions=schema,report={**report,'authored_contrast_families':assignment['components'],
            'partial_advice':sum(r['labels']['D2'] and not r['metadata']['source_refs'][0]['is_final'] for r in rows),
            'partial_no_advice':sum(not r['labels']['D2'] and not r['metadata']['source_refs'][0]['is_final'] for r in rows)},dataset_id=digest({'rows':rows}))
        page=(root/'scripts/templates/streaming_review_v2.html').read_text().replace('__DATASET_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
        page=page.replace('Version 2','Context v5 construction release').replace('authored trajectories','constructed trajectories').replace('End of authored turn','End of constructed turn')
        (stage/'review.html').write_text(page)
        sources=['src/decision_guardrail/data_generation/context_v5_release.py','src/decision_guardrail/data_generation/context_v5_content.py',
            'data/authored/streaming_context_v5_release/source_label_corrections.json','data/authored/streaming_context_v5/seed_families.py',
            'config/dataset/question_schema_streaming_v5.yaml','docs/annotation_standard_streaming_v5.md',
            'scripts/templates/streaming_review_v2.html','src/decision_guardrail/data_generation/streaming_15k_catalog.py',
            'src/decision_guardrail/data_generation/streaming_15k_v4_content.py','data/generated/streaming_15000_v4/manifest.json']
        manifest=dict(schema_version=VERSION,generator=GENERATOR,status=STATUS,records=len(rows),scenarios=len(trajectories),
            independent_annotations=0,source_master_sha256=sha(root/'data/generated/streaming_15000_v4/records.jsonl'),
            source_hashes={name:sha(root/name) for name in sources},
            files={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()})
        write_json(stage/'manifest.json',manifest);stage.rename(output)
    return report


def verify(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['schema_version']!=VERSION or manifest['generator']!=GENERATOR:raise ValueError('Wrong release contract')
    for name,expected in manifest['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p)!=expected:raise ValueError('Artifact hash mismatch: '+name)
    rows=[json.loads(l) for l in (output/'records.jsonl').read_text().splitlines()]
    trajectories=trajectories_for(rows);actual=quality(rows,trajectories)
    saved=json.loads((output/'quality_report.json').read_text())
    if manifest['records']!=len(rows) or manifest['scenarios']!=len(trajectories):raise ValueError('Manifest count mismatch')
    for key,value in actual.items():
        # JSON serialisation turns integer count keys into strings.
        if json.loads(json.dumps(value))!=saved[key]:raise ValueError('Report differs: '+key)
    if lexical_candidates(rows):raise ValueError('Unresolved lexical candidates')
    qs=json.loads((output/'native_questions.json').read_text());idx={r['id']:i for i,r in enumerate(rows,1)}
    for split in SPLITS:
        selected=[r for r in rows if r['metadata']['split']==split]
        native=[json.loads(l) for l in (output/'clef'/f'{split}.jsonl').read_text().splitlines()]
        split_rows=[json.loads(l) for l in (output/'splits'/f'{split}.jsonl').read_text().splitlines()]
        refs=[json.loads(l) for l in (output/'clef_index'/f'{split}.jsonl').read_text().splitlines()]
        if split_rows!=selected or len(native)!=len(selected) or len(refs)!=len(selected):raise ValueError('Export count/parity mismatch')
        for i,(r,n,ref) in enumerate(zip(selected,native,refs,strict=True),1):
            if n!=dict(state=state_text(**r['input']),questions=qs,gold=r['labels']):raise ValueError('Native export differs')
            if ref!=dict(export_line=i,record_id=r['id'],records_jsonl_line=idx[r['id']],recommendation_withdrawn_in_target=r['metadata']['recommendation_withdrawn_in_target']):raise ValueError('Native lineage differs')
    return actual
