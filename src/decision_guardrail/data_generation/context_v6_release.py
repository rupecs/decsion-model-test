"""Remediate Claude X01–X06 in a new immutable, explicitly non-trainable release."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import yaml
from .context_v5_review import sha, digest, write_json, write_rows
from .context_v5_release import lexical_candidates, trajectories_for
from .streaming_15k_catalog import TOPICS
from ..dataset import state_text, validate_record
from ..schema import CLASSES, decision_violations

VERSION='banking_streaming_d1_d7_v6_context_draft'
GENERATOR='streaming-context-remediation-v6'
SPLITS=('train','validation','calibration','test')
ROOT=Path(__file__).resolve().parents[3]


def native_questions(root):
    schema=yaml.safe_load((root/'config/dataset/question_schema_streaming_v6.yaml').read_text())
    qs={d:dict(type='noul',instructions=q['instructions']) for d,q in schema['decisions'].items() if d!='D7'}
    qs['D7']=dict(type='choice',instructions=schema['decisions']['D7']['instructions'],criteria={
        'factual_information':'Substantive product facts/procedure without endorsed advice in this target.',
        'general_advice':'Endorsed product advice without actual-basis use or individual suitability in context.',
        'personal_advice':'Product advice demonstrably using a substantive actual basis or asserting/implying individual suitability.',
        'other':'No substantive facts, procedure or endorsed product advice expressed so far.'})
    return schema,qs


def normal_input(inp):
    return tuple(' '.join(re.findall(r"[a-z0-9']+",inp[k].casefold())) for k in ('customer_context','agent_response'))


def close_units(rows,units):
    units=set(units)
    while True:
        linked={u for r in rows if units.intersection(r['metadata']['units']) for u in r['metadata']['units']}
        if linked<=units:return units
        units|=linked


def prepare_source(root):
    source=root/'data/generated/streaming_context_v5_release/records.jsonl'
    originals=[json.loads(line) for line in source.read_text().splitlines()]
    review=root/'data/reviews/claude_streaming_context_v5_review'
    sample=[json.loads(line) for line in (review/'record_review.jsonl').read_text().splitlines()]
    sample_ids={r.get('record_id',r.get('id')) for r in sample}
    sample_ids.discard(None)
    if not sample_ids or not sample_ids<=set(r['id'] for r in originals):raise ValueError('Unknown Claude sample IDs')
    ledger=json.loads((root/'data/authored/streaming_context_v6/D3_dispositions.json').read_text())
    decisions={r['source_id']:r for r in ledger}
    by_id={r['id']:r for r in originals}
    exposed_units={u for r in originals if r['id'] in sample_ids for u in r['metadata']['units']}
    exposed_units.update(u for r in originals for u in r['metadata']['units'] if u.startswith('addition:review-'))
    exposed_units=close_units(originals,exposed_units)
    uncertain_units={u for d in ledger if d['action']=='quarantine' for u in by_id[d['source_id']]['metadata']['units']}
    uncertain_units=close_units(originals,uncertain_units)
    rows=[];exposed=[];quarantine=[];changes=[];dispositions=[]
    for line,source_row in enumerate(originals,1):
        r=deepcopy(source_row);m=r['metadata'];decision=decisions.get(r['id'])
        if decision and (r['input']!=decision['expected_input'] or r['labels']!=decision['before_labels']):raise ValueError('D3 ledger precondition changed')
        disposition='keep';why='Preserved construction labels; independent annotation remains pending.'
        if decision and decision['action']=='correct_D3_false':
            if not r['labels']['D4']:raise ValueError('Expected D4 to preserve personal classification')
            r['labels']['D3']=False;r['annotation']['rationale']=decision['rationale']
            r['annotation']['profile']='individual_suitability_information_basis_only'
            m['original_customer_basis_type']=m.get('customer_basis_type')
            m['customer_basis_type']='information_request'
            changes.append(dict(source_id=source_row['id'],before=source_row['labels'],after=deepcopy(r['labels']),reason=decision['rationale']))
            disposition='correct_D3';why=decision['rationale']
        elif decision:why=decision['rationale']
        m['v5_source_id']=source_row['id'];m['v5_original_split']=m['split']
        m['lineage'].append(dict(version=source_row['schema_version'],source_id=source_row['id'],source_line=line,source_sha256=digest(source_row)))
        r['id']='sctxv6-'+digest(r['input'])[:24];r['schema_version']=VERSION
        r['annotation'].update(rule_version=VERSION,status='pre_annotation',independent_review=False)
        if exposed_units.intersection(m['units']):
            m.update(split=None,native_export_eligible=False);disposition='exposed_development_only'
            exposed.append(r)
        elif uncertain_units.intersection(m['units']):
            m.update(split=None,native_export_eligible=False);disposition='quarantine_boundary'
            quarantine.append(r)
        else:
            m['native_export_eligible']=True;rows.append(r)
        dispositions.append(dict(source_id=source_row['id'],source_line=line,disposition=disposition,rationale=why,
            labels_changed=source_row['labels']!=r['labels'],independent_adjudication=False))
    return rows,exposed,quarantine,changes,dispositions,ledger


HEADERS={
 'train':('Your statement was "{quote}". ', 'I acknowledge your statement, "{quote}". '),
 'validation':('I have recorded your statement as "{quote}"; that records the request rather than assessing it. ',),
 'calibration':('The note of your statement reads "{quote}". Recording it does not determine a selection. ',),
 'test':('The words in your statement are "{quote}". Acknowledgement alone is not a product assessment. ',),
}
PERSONAL=(
 '{P} is the appropriate choice for you.',
 'For the current customer, I would select {p}.',
 'My individual selection for this situation is {p}.',
 '{P} is what I would choose on your behalf.',
 'The personally suitable option here is {p}.',
 'I suggest you opt for {p}.',
)


def labels_for(kind,credit):
    bits={'factual_information':(1,0,0,0,1,0),'general_advice':(1,1,0,0,0,credit),
          'personal_advice':(1,1,0,1,0,credit),'other':(1,0,0,0,0,0)}[kind]
    return {**{f'D{i}':bool(v) for i,v in enumerate(bits,1)},'D7':kind}


def target_for(context,topic,kind,variant,split):
    # Quote the actual first CUSTOMER turn; no invented financial situation.
    quote=next(line.removeprefix('CUSTOMER: ') for line in context.splitlines() if line.startswith('CUSTOMER: '))
    quote=quote.replace('"',"'")
    headers=HEADERS[split];header=headers[variant%len(headers)].format(quote=quote)
    p=topic.product;P=p[0].upper()+p[1:]
    v=variant//len(headers)
    if kind=='factual_information':
        body=(topic.fact,topic.condition,f'The document for {p} contains its product conditions.',
              topic.fact+' '+topic.condition,
              f'The feature section of the terms for {p} can be located through the document index.',
              topic.condition+f' The conditions section of the document for {p} supplies the applicable wording.')[v%6]
    elif kind=='general_advice':
        opinion=(topic.preference,topic.caution,topic.preference+' '+topic.caution)[v%3]
        body=('Separately, my general product-feature opinion is: '+opinion if v%6<3 else
              'As a comparison of features rather than an individual selection: '+opinion)
    elif kind=='personal_advice':body=PERSONAL[v%len(PERSONAL)].format(p=p,P=P)
    else:
        body=(f'I am not offering product advice about {p}.',
              f'For {p}, I am withholding a product selection.',
              f'About {p}: no product advice is being given in this reply.',
              f'The quoted view is "Choose {p}"; I am quoting it without endorsement.')[v%4]
    return header+body


def counterexamples(rows,blocked_inputs):
    topics={t.key:t for t in TOPICS};contexts=defaultdict(list);quote_contexts={}
    for r in rows:
        contexts[r['input']['customer_context']].append(r)
        if re.search(r'\byour statement\b',r['input']['agent_response'],re.I):quote_contexts[r['input']['customer_context']]=r
    selected={ctx:members for ctx,members in contexts.items() if any(r['metadata']['origin']!='retained_v4' for r in members)}
    existing={normal_input(r['input']) for r in rows};added=[];designs=[]
    def add(ctx,reference,kind,number,design):
        topic=topics[reference['metadata']['parent_topic']];split=reference['metadata']['split']
        for variant in range(number,96):
            text=target_for(ctx,topic,kind,variant,split);inp=dict(customer_context=ctx,agent_response=text)
            key=normal_input(inp)
            if key in existing or key in blocked_inputs:continue
            existing.add(key);unit='v6-cross:'+digest({'context':ctx})[:20]
            scenario=unit+':'+kind+':'+str(variant)
            meta=deepcopy(reference['metadata'])
            meta.update(origin='v6_crossed_response_design',units=[unit],source_ids=[],
                source_refs=[dict(scenario_id=scenario,checkpoint_index=0,is_final=True,total_checkpoints=1,value_added=design)],
                construction_recipe=design,coverage_boundary=design,recommendation_withdrawn_in_target=False,recommendation_withdrawn=False,
                native_export_eligible=True,lineage=[dict(version=VERSION,parent_context_source=reference['metadata']['v5_source_id'],design=design,variant=variant)],
                v5_source_id=None)
            labels=labels_for(kind,topic.credit)
            r=dict(id='sctxv6-'+digest(inp)[:24],schema_version=VERSION,input=inp,labels=labels,advice_present=labels['D2'],
                annotation=dict(profile=kind,rationale='The quoted request is acknowledged, not used as an advice basis. Classify the subsequent speech act; a generic request cannot override an individual selection.',
                    rule_version=VERSION,status='pre_annotation',independent_review=False,visible_evidence=text,
                    evidence=dict(target_spans=[dict(start=0,end=len(text),text=text)],context_turns=[dict(id=f'context-{i}',text=line) for i,line in enumerate(ctx.splitlines(),1)])),metadata=meta)
            added.append(r);return r
        raise ValueError('No distinct meaningful counterexample remains; do not pad a quota')
    for ctx,members in sorted(selected.items()):
        counts=Counter(r['labels']['D7'] for r in members);target=max(counts.values());reference=members[0]
        # Balanced factorial response coverage at an identical context; no labels
        # are predicted from the context or assigned from keyword matches.
        for kind in CLASSES:
            for i in range(target-counts[kind]):add(ctx,reference,kind,i,'same_context_four_speech_acts')
        designs.append(dict(context_sha256=digest({'context':ctx}),existing_counts=dict(counts),target_per_class=target))
    # Cross customer quotation with all four speech acts in the *same* context.
    for ctx,reference in sorted(quote_contexts.items()):
        for kind in CLASSES:add(ctx,reference,kind,20,'customer_quote_four_speech_acts')
    return rows+added,dict(balanced_addition_contexts=len(selected),quote_contexts=len(quote_contexts),new_counterexamples=len(added),designs=designs)


def cue_diagnostics(rows):
    contexts=defaultdict(Counter)
    for r in rows:contexts[r['input']['customer_context']][r['labels']['D7']]+=1
    counts=defaultdict(Counter)
    for ctx,classes in contexts.items():
        words=re.findall(r"[a-z0-9']+",ctx.casefold())
        grams={tuple(words[i:i+n]) for n in range(1,6) for i in range(len(words)-n+1)}
        for gram in grams:counts[gram].update(classes)
    cues=[]
    for gram,dist in counts.items():
        n=sum(dist.values());winner,number=dist.most_common(1)[0]
        if n>=30 and number/n>.8:cues.append(dict(phrase=' '.join(gram),support=n,dominant_class=winner,precision=number/n))
    response={}
    for phrase in ('your statement','advice'):
        dist=Counter(r['labels']['D7'] for r in rows if re.search(r'\b'+re.escape(phrase)+r'\b',r['input']['agent_response'],re.I))
        response[phrase]=dict(counts=dict(dist),support=sum(dist.values()),personal_precision=dist['personal_advice']/sum(dist.values()) if dist else None)
    return dict(context_cues_support30_precision80=len(cues),context_cues=sorted(cues,key=lambda r:(-r['precision'],-r['support']))[:100],response_cues=response,
        method='Document-frequency word ngrams 1–5; context support>=30, class precision>0.8. Diagnostic gates do not establish generalisation.')


def validate_rows(rows,exposed,qs):
    seen=set();units=defaultdict(set);parents=defaultdict(set)
    forbidden={normal_input(r['input']) for r in exposed}
    for r in rows:
        if r['schema_version']!=VERSION or decision_violations(r['labels']):raise ValueError('Invalid candidate contract')
        key=normal_input(r['input'])
        if key in forbidden or key in seen:raise ValueError('Exposed or duplicate full input in candidate partitions')
        seen.add(key);m=r['metadata']
        if any(u.startswith('addition:review-') for u in m['units']):raise ValueError('Exposed review family leaked')
        for u in m['units']:units[u].add(m['split'])
        parents[m['parent_topic']].add(m['split'])
        validate_record(dict(state=state_text(**r['input']),questions=qs,gold=r['labels']),list(CLASSES))
    if any(len(s)>1 for s in [*units.values(),*parents.values()]):raise ValueError('Linked unit crosses candidate splits')
    trajectories=trajectories_for(rows)
    report=cue_diagnostics(rows)
    if report['context_cues_support30_precision80']:raise ValueError('Customer-context shortcut gate failed: '+str(report['context_cues'][:5]))
    if any(v['personal_precision']>.8 for v in report['response_cues'].values() if v['support']>=30):raise ValueError('Customer quotation/advice remains a personal-only cue')
    report.update(records=len(rows),scenarios=len(trajectories),split_counts={s:sum(r['metadata']['split']==s for r in rows) for s in SPLITS},
        class_counts=dict(Counter(r['labels']['D7'] for r in rows)),approved_training_records=0,independently_adjudicated_records=0,
        training_ready=False,native_token_coverage='not_run',exposed_inputs_in_candidate_partitions=0,linked_units_crossing_splits=0)
    return report,trajectories


def build(root,output):
    if output.exists():raise ValueError('Choose a new output path; preserve previous releases')
    rows,exposed,quarantine,changes,dispositions,ledger=prepare_source(root)
    schema,qs=native_questions(root)
    rows,cross=counterexamples(rows,{normal_input(r['input']) for r in exposed+quarantine})
    # Do not silently discard counterexamples and reintroduce the context cue.
    if lexical_candidates(rows):raise ValueError('Cross-partition lexical candidates require design remediation before release')
    report,trajectories=validate_rows(rows,exposed,qs)
    report.update(counterexample_design={k:v for k,v in cross.items() if k!='designs'},exposed_development_records=len(exposed),
        quarantined_records=len(quarantine),D3_ledger_counts=dict(Counter(d['action'] for d in ledger)),
        D3_corrected_candidate_rows=sum(r['metadata'].get('v5_source_id') in {c['source_id'] for c in changes} for r in rows),
        construction_export_only=True,limitations=['Pre-annotation labels; zero approved training exports.',
          'Model-review samples and original review families are exposed development only.',
          'Matched counterexamples and ngram gates do not establish independent model generalisation.',
          'Native encoding and contract-specific GPU smoke/reload remain pending; no GPU was started.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='context-v6-',dir=output.parent) as tmp:
        stage=Path(tmp)/'release';stage.mkdir()
        write_rows(stage/'records.jsonl',rows);write_json(stage/'quality_report.json',report);write_json(stage/'trajectories.json',trajectories)
        write_json(stage/'questions.json',schema);write_json(stage/'native_questions.json',qs);write_json(stage/'counterexample_design.json',cross)
        index={r['id']:i for i,r in enumerate(rows,1)};write_json(stage/'line_index.json',index)
        for name in ('splits','construction_clef','clef_index','exposed_development'):(stage/name).mkdir()
        for split in SPLITS:
            selected=[r for r in rows if r['metadata']['split']==split]
            write_rows(stage/'splits'/f'{split}.jsonl',selected)
            write_rows(stage/'construction_clef'/f'{split}.jsonl',[dict(state=state_text(**r['input']),questions=qs,gold=r['labels']) for r in selected])
            write_rows(stage/'clef_index'/f'{split}.jsonl',[dict(export_line=i,record_id=r['id'],records_jsonl_line=index[r['id']],recommendation_withdrawn_in_target=r['metadata']['recommendation_withdrawn_in_target']) for i,r in enumerate(selected,1)])
        write_rows(stage/'exposed_development'/'records.jsonl',exposed);write_rows(stage/'quarantine.jsonl',quarantine)
        write_rows(stage/'source_dispositions.jsonl',dispositions);write_rows(stage/'label_changes.jsonl',changes)
        write_json(stage/'D3_adjudication_queue.json',[d for d in ledger if d['action']=='quarantine'])
        write_json(stage/'annotation_ledger.json',[])
        write_rows(stage/'review_queue.jsonl',[dict(record_id=r['id'],input_labels_sha256=digest(dict(input=r['input'],labels=r['labels'])),status='pending_independent_annotation') for r in rows])
        (stage/'annotation_standard.md').write_text((root/'docs/annotation_standard_streaming_v6.md').read_text())
        (stage/'README.md').write_text('# Context v6 pre-annotation remediation\n\n'+f'{len(rows)} candidate snapshots; zero approved training rows.\n'+
            'construction_clef/ contains native-format CPU/review fixtures only. There is no approved clef/ directory.\n'+
            'training_ready=false is enforced before GPU access. Exposed review/sample families have split=null under exposed_development/.\n'+
            'See quality_report.json, label_changes.jsonl, D3_adjudication_queue.json and annotation_standard.md.\n')
        # The review page needs exact input/labels and navigation, not the full
        # repeated audit ledger. Intern contexts to stay within browser IPC limits.
        contexts=list(dict.fromkeys(r['input']['customer_context'] for r in rows))
        context_index={text:i for i,text in enumerate(contexts)}
        review_rows=[dict(id=r['id'],input=dict(customer_context=context_index[r['input']['customer_context']],agent_response=r['input']['agent_response']),
            labels=r['labels'],advice_present=r['advice_present'],annotation=dict(rationale=r['annotation']['rationale']),
            metadata={key:r['metadata'][key] for key in ('contrast_family','split','recommendation_withdrawn','coverage_boundary','source_refs')}) for r in rows]
        payload=dict(rows=review_rows,contexts=contexts,questions=schema,report={**report,'authored_contrast_families':len({r['metadata']['split_group'] for r in rows}),
            'credit_advice_rows':sum(r['labels']['D6'] for r in rows)},dataset_id=digest({'rows':rows}))
        page=(root/'scripts/templates/streaming_review_v2.html').read_text().replace('__DATASET_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
        page=page.replace('const data=JSON.parse(document.getElementById(\'dataset\').textContent), rows=data.rows;',
            'const data=JSON.parse(document.getElementById(\'dataset\').textContent), rows=data.rows;\n'
            'rows.forEach(row=>{row.input.customer_context=data.contexts[row.input.customer_context];row.annotation.visible_evidence=row.input.agent_response;});')
        page=page.replace('Version 2','Context v6 PRE-ANNOTATION · TRAINING BLOCKED').replace('contrast families','linked split groups').replace('authored trajectories','constructed trajectories').replace('End of authored turn','End of constructed turn')
        (stage/'review.html').write_text(page)
        sources=['src/decision_guardrail/data_generation/context_v6_release.py','data/authored/streaming_context_v6/D3_dispositions.json',
            'docs/annotation_standard_streaming_v6.md','config/dataset/question_schema_streaming_v6.yaml',
            'data/generated/streaming_context_v5_release/manifest.json','data/reviews/claude_streaming_context_v5_review/affected_ids.json',
            'data/reviews/claude_streaming_context_v5_review/record_review.jsonl',
            'data/generated/streaming_context_v5_release/records.jsonl',
            'src/decision_guardrail/training_readiness.py','src/decision_guardrail/trainer.py',
            'src/decision_guardrail/schema.py','src/decision_guardrail/streaming_data.py',
            'src/decision_guardrail/dataset.py','src/decision_guardrail/preparation.py',
            'config/dataset/question_schema_runtime_v6.yaml',
            'src/decision_guardrail/data_generation/context_v5_release.py',
            'src/decision_guardrail/data_generation/context_v5_review.py',
            'src/decision_guardrail/data_generation/streaming_15k_catalog.py']
        manifest=dict(schema_version=VERSION,generator=GENERATOR,status='pre_annotation_not_trainable',training_ready=False,
            native_export_directory='construction_clef',annotation_ledger_path='annotation_ledger.json',records=len(rows),scenarios=len(trajectories),
            source_hashes={s:sha(root/s) for s in sources},files={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()})
        write_json(stage/'manifest.json',manifest);stage.rename(output)
    return report


def verify(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['schema_version']!=VERSION or manifest.get('training_ready') is not False or manifest['native_export_directory']!='construction_clef':raise ValueError('Wrong pre-annotation manifest')
    for name,expected in manifest['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p)!=expected:raise ValueError('Artifact hash mismatch: '+name)
    if (output/'clef').exists():raise ValueError('Unapproved native export directory exists')
    rows=[json.loads(l) for l in (output/'records.jsonl').read_text().splitlines()]
    exposed=[json.loads(l) for l in (output/'exposed_development/records.jsonl').read_text().splitlines()]
    qs=json.loads((output/'native_questions.json').read_text());report,_=validate_rows(rows,exposed,qs)
    saved=json.loads((output/'quality_report.json').read_text())
    for key,value in report.items():
        if value!=saved[key]:raise ValueError('Quality report differs: '+key)
    if lexical_candidates(rows):raise ValueError('Unresolved cross-partition lexical overlap')
    for split in SPLITS:
        selected=[r for r in rows if r['metadata']['split']==split]
        exported=[json.loads(l) for l in (output/'construction_clef'/f'{split}.jsonl').read_text().splitlines()]
        if exported!=[dict(state=state_text(**r['input']),questions=qs,gold=r['labels']) for r in selected]:raise ValueError('Construction export differs from master')
    return report
