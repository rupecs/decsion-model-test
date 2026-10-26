"""Full construction replacement after the v5–v7 reviews; never training approval."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import runpy
import tempfile

from .context_v5_review import digest, sha, write_json, write_rows
from .context_v5_release import labels_for, lexical_candidates
from .context_v6_release import VERSION, SPLITS, close_units, native_questions, normal_input, validate_rows
from .natural_context_review import duplicate_bodies, full_turn_echoes
from ..dataset import state_text
from ..training_readiness import annotation_fingerprint

GENERATOR = 'streaming-context-replacement-v8'
SOURCE = 'data/generated/streaming_context_v6_release_r3'
EMBARGO = 'data/generated/streaming_context_v7_coherent_development/exposed_v6_review_units.json'
WITHDRAWALS = 'data/generated/streaming_context_v7_coherent_development/withdrawn_v6_counterexamples.jsonl'
FAMILIES = 'data/authored/streaming_context_v8/families.json'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_files(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError('Artifact hash mismatch: '+name)
    return manifest


def repaired_development(root):
    original = read_rows(root/'data/generated/streaming_context_v7_development/records.jsonl')
    flagged = set(json.loads((root/'data/reviews/claude_streaming_context_v7_review/affected_ids.json').read_text())['Z03_reply_references_absent_context'])
    if len(flagged) != 49 or not flagged <= {r['id'] for r in original}:
        raise ValueError('Unexpected development repair scope')
    replace = runpy.run_path(str(root/'data/authored/streaming_context_v8/development_repairs.py'))['replacement']
    revised, changes = [], []
    for source in original:
        row = deepcopy(source)
        row['metadata'].update(split=None, native_export_eligible=False, exposed_development_only=True,
                               source_record_id=source['id'])
        if source['id'] in flagged:
            row['input']['agent_response'] = replace(source)
            if row['input'] == source['input']:
                raise ValueError('A flagged development cell was not repaired')
            row['id'] = 'v8-development-'+digest(row['input'])[:24]
            row['annotation']['rationale'] = ('Context-specific replacement: individual fit asserts no invented circumstances; any explicit basis uses only the circumstances in this context. Proposed labels require independent annotation.')
            changes.append(dict(source_id=source['id'],replacement_id=row['id'],before_input=source['input'],after_input=row['input'],
                                before_labels=source['labels'],after_labels=row['labels'],independent_annotation=False))
        row['annotation'].update(status='pending_independent_annotation', independent_review=False)
        revised.append(row)
    if len(changes) != 49 or duplicate_bodies(revised) or full_turn_echoes(revised):
        raise ValueError('Development repair integrity failed')
    return revised, changes, original


def prepare_source(root):
    source = root/SOURCE
    verify_files(source)
    rows = read_rows(source/'records.jsonl')
    withdrawn = {r['record_id'] for r in read_rows(root/WITHDRAWALS)}
    embargo = json.loads((root/EMBARGO).read_text())
    exposed_ids = set(embargo['record_ids'])
    if withdrawn != {r['id'] for r in rows if r['metadata']['origin']=='v6_crossed_response_design'} or len(withdrawn)!=3097:
        raise ValueError('Withdrawal ledger differs from frozen v6 designs')
    if len(exposed_ids)!=121 or not exposed_ids <= {r['id'] for r in rows}:
        raise ValueError('Invalid v6 exposure embargo')
    exposed_units = close_units(rows, set(embargo['units']))
    if {r['id'] for r in rows if set(r['metadata']['units']) & exposed_units} != exposed_ids:
        raise ValueError('Exposure embargo is not closed over linked units')
    eligible = [r for r in rows if r['metadata']['origin']=='retained_v4' and r['id'] not in exposed_ids]
    quote_units = close_units(eligible, {u for r in eligible if 'your statement' in r['input']['agent_response'].casefold() for u in r['metadata']['units']})
    retained, quarantine, exposure, dispositions = [], read_rows(source/'quarantine.jsonl'), read_rows(source/'exposed_development/records.jsonl'), []
    owners = {}
    for source_row in rows:
        m = source_row['metadata']; topic = m['parent_topic']
        owner = (m['split'], m['split_group'])
        if topic in owners and owners[topic] != owner:
            raise ValueError('Source topic has multiple partition owners')
        owners[topic] = owner
        reasons = []
        if source_row['id'] in withdrawn:reasons.append('withdrawn_v6_crossed_design')
        if source_row['id'] in exposed_ids:reasons.append('review_exposure_embargo')
        if m['origin']=='compositional_expansion':reasons.append('superseded_v5_addition_recipe')
        if m['origin']=='retained_v4' and set(m['units']) & quote_units:reasons.append('quarantine_legacy_quote_scaffold_unit')
        row = deepcopy(source_row)
        if reasons:
            row['metadata'].update(split=None,native_export_eligible=False,exclusion_reasons=reasons)
            if 'review_exposure_embargo' in reasons:exposure.append(row)
            elif 'quarantine_legacy_quote_scaffold_unit' in reasons:quarantine.append(row)
        else:
            if m['origin'] != 'retained_v4':raise ValueError('Unknown source origin')
            row['id'] = 'sctxv8-'+digest(row['input'])[:24]
            row['metadata']['v6_source_id'] = source_row['id']
            row['metadata']['lineage'].append(dict(version=GENERATOR,source_id=source_row['id'],source_record_sha256=digest(source_row)))
            row['annotation'].update(status='pre_annotation',independent_review=False,rule_version=GENERATOR)
            retained.append(row)
        dispositions.append(dict(source_id=source_row['id'],source_sha256=digest(source_row),
            disposition=reasons[0] if reasons else 'retain',all_exclusion_reasons=reasons,
            candidate_id=None if reasons else row['id'],labels_changed=False))
    return retained, exposure, quarantine, dispositions, owners


def authored_rows(root, owners):
    families = json.loads((root/FAMILIES).read_text())
    if len(families)!=60 or {f['id'] for f in families}!=set(owners):
        raise ValueError('All 60 topic designs must have a frozen partition owner')
    by_input = {}; support_ledger = []
    for family in families:
        topic = family['id']; split, group = owners[topic]
        unit = 'v8-natural:'+topic
        if len({p['id'] for p in family['points']}) != len(family['points']):raise ValueError('Duplicate authored point ID')
        for p in family['points']:
            if p['context'] != family['contexts'][p['context_id']]:raise ValueError('Context and explicit cell disagree')
            evidence = p['available_context_evidence']
            if not set(p['requires']) <= set(evidence):raise ValueError('Reply requires absent context facts')
            spans = []
            for tag in p['requires']:
                text = evidence[tag]; start = p['context'].find(text)
                if not text or start<0:raise ValueError('Declared context evidence does not occur in this conversation')
                spans.append(dict(tag=tag,start=start,end=start+len(text),text=text))
            if p['profile']=='linked' and 'actual_need' not in p['requires']:
                raise ValueError('An explicit-basis reply needs a recorded actual-need witness')
            labels = labels_for(p['profile'],family['credit'])
            prefix = p['prefix']; checkpoint = prefix['checkpoint_index'] if prefix else 0
            scenario = unit+':'+('prefix' if prefix else p['id'])
            ref = dict(scenario_id=scenario,checkpoint_index=checkpoint,total_checkpoints=4 if prefix else 1,
                       is_final=prefix['is_final'] if prefix else True,value_added=family['design'])
            inp = dict(customer_context=p['context'],agent_response=p['text'])
            row = dict(id='sctxv8-'+digest(inp)[:24],schema_version=VERSION,input=inp,labels=labels,advice_present=labels['D2'],
                annotation=dict(profile=p['profile'],rationale=p['rationale'],rule_version=GENERATOR,status='pre_annotation',
                    independent_review=False,author=family['author'],visible_evidence=p['text'],
                    evidence=dict(context_support_spans=spans,target_spans=[dict(start=0,end=len(p['text']),text=p['text'])])),
                metadata=dict(parent_topic=topic,origin='v8_natural_context_design',units=[unit],source_ids=[],source_refs=[ref],
                    split=split,original_split=None,split_group=group,contrast_family=group,
                    customer_situation_id='ctx-'+digest({'context':p['context']})[:20],
                    scenario_product_domains=[family['domain']],customer_basis_type='information_request' if p['context_id']=='generic' else 'authored_actual_need',
                    customer_disclosure=p['context_id']!='generic',construction_recipe=p['context_id'],coverage_boundary='explicit_natural_context_cells',
                    recommendation_withdrawn_in_target=bool(prefix and prefix['is_final']),recommendation_withdrawn=bool(prefix and prefix['is_final']),
                    native_export_eligible=True,synthetic=True,independent_annotation=False,v5_source_id=None,v6_source_id=None,
                    lineage=[dict(version=GENERATOR,authored_family=topic,authored_point=p['id'])]))
            key = normal_input(inp)
            if key in by_input:
                existing = by_input[key]
                if existing['input']!=inp or existing['labels']!=labels or existing['metadata']['units']!=[unit]:
                    raise ValueError('Authored merge would hide a conflicting input')
                existing['metadata']['source_refs'].append(ref)
                existing['metadata']['lineage'].extend(row['metadata']['lineage'])
            else:by_input[key]=row
            support_ledger.append(dict(family=topic,point=p['id'],record_id=row['id'],context_id=p['context_id'],
                                       required_context_tags=p['requires'],context_support_spans=spans))
    return list(by_input.values()), support_ledger


def response_cues(rows):
    counts = defaultdict(Counter)
    for r in rows:
        words = re.findall(r"[a-z0-9']+", r['input']['agent_response'].casefold())
        for gram in {tuple(words[i:i+n]) for n in range(1,6) for i in range(len(words)-n+1)}:
            counts[gram][r['labels']['D7']]+=1
    cues=[]
    for gram,dist in counts.items():
        support=sum(dist.values()); winner,number=dist.most_common(1)[0]
        if support>=100 and number/support>.95:
            cues.append(dict(phrase=' '.join(gram),support=support,dominant_class=winner,precision=number/support))
    return sorted(cues,key=lambda c:(-c['precision'],-c['support'],c['phrase']))


def quality(rows, exposed, qs):
    report, trajectories = validate_rows(rows,exposed,qs)
    if duplicate_bodies(rows):raise ValueError('Same-context reply bodies are duplicated')
    if full_turn_echoes(rows):raise ValueError('A reply copies a full customer turn')
    overlaps=lexical_candidates(rows)
    if overlaps:
        raise ValueError('Cross-partition lexical overlap: '+str([(a['metadata']['parent_topic'],b['metadata']['parent_topic'],s) for a,b,s in overlaps[:5]]))
    natural=defaultdict(set)
    for r in rows:
        if r['metadata']['origin']=='v8_natural_context_design':
            natural[(r['metadata']['parent_topic'],r['metadata']['construction_recipe'])].add(r['labels']['D7'])
    if len(natural)!=60*5 or any(classes!={'factual_information','general_advice','personal_advice','other'} for classes in natural.values()):
        raise ValueError('A natural context has lost one of the four speech-act contrasts')
    report.update(natural_context_groups_with_all_four_classes=len(natural),same_context_duplicate_bodies=0,full_turn_echoes=0,cross_partition_complete_target_5gram_jaccard65=0,
        class_counts_by_split={s:dict(Counter(r['labels']['D7'] for r in rows if r['metadata']['split']==s)) for s in SPLITS},
        binary_counts={d:dict(Counter(str(r['labels'][d]).lower() for r in rows)) for d in ('D1','D2','D3','D4','D5','D6')},
        distinct_replies=len({r['input']['agent_response'] for r in rows}),
        natural_candidate_rows=sum(r['metadata']['origin']=='v8_natural_context_design' for r in rows),
        retained_legacy_rows=sum(r['metadata']['origin']=='retained_v4' for r in rows))
    return report,trajectories


def build(root, output):
    if output.exists():raise ValueError('Choose a new output path; preserve all frozen releases')
    retained,exposed,quarantine,dispositions,owners=prepare_source(root)
    new,support=authored_rows(root,owners)
    development,repairs,original_development=repaired_development(root)
    for row in exposed+original_development+development:
        row['metadata'].update(split=None,native_export_eligible=False,exposed_development_only=True)
    # Identical exposed inputs are one development item with no candidate split.
    exposure_map={normal_input(r['input']):r for r in exposed+original_development+development}
    exposed=list(exposure_map.values())
    rows=retained+new
    if {normal_input(r['input']) for r in quarantine}&{normal_input(r['input']) for r in rows}:
        raise ValueError('Quarantined source was promoted')
    schema,qs=native_questions(root)
    report,trajectories=quality(rows,exposed,qs)
    cues=response_cues(rows)
    report.update(status='full_construction_replacement_pending_independent_annotation',generator=GENERATOR,
        schema_reused=VERSION,topic_families=60,authored_design_points=len(support),
        shared_authored_prefix_starts=len(support)-len(new),source_disposition_counts=dict(Counter(d['disposition'] for d in dispositions)),
        exposure_embargo_records=121,withdrawn_designs=3097,exposed_development_records=len(exposed),
        quarantined_records=len(quarantine),repaired_development_rows=len(repairs),
        response_cues_support100_precision95=len(cues),construction_export_only=True,
        limitations=['All candidate labels remain construction proposals; zero independent annotation approvals.',
                    'Synthetic shared-recipe candidate partitions are not independent challenge validation or legal approval.',
                    'Evidence spans and declared support tags are mechanical checks, not semantic entailment or human coherence approval.',
                    'Response ngram diagnostics include legitimate class semantics and do not prove shortcut elimination.',
                    'Native token coverage and current-contract GPU smoke/save/offline-reload are pending; no GPU was started.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='context-v8-',dir=output.parent) as tmp:
        stage=Path(tmp)/'release';stage.mkdir()
        for directory in ('splits','construction_clef','clef_index','exposed_development'):(stage/directory).mkdir()
        write_rows(stage/'records.jsonl',rows);write_json(stage/'quality_report.json',report)
        write_json(stage/'questions.json',schema);write_json(stage/'native_questions.json',qs)
        write_json(stage/'trajectories.json',trajectories);write_rows(stage/'source_dispositions.jsonl',dispositions)
        write_rows(stage/'quarantine.jsonl',quarantine);write_rows(stage/'exposed_development/records.jsonl',exposed)
        write_rows(stage/'exposed_development/repaired_v7_records.jsonl',development)
        write_rows(stage/'exposed_development/v7_repair_changes.jsonl',repairs)
        write_rows(stage/'context_support_ledger.jsonl',support);write_json(stage/'response_cue_audit.json',cues)
        (stage/'withdrawal_ledger.jsonl').write_bytes((root/WITHDRAWALS).read_bytes())
        (stage/'exposure_embargo.json').write_bytes((root/EMBARGO).read_bytes())
        write_json(stage/'annotation_ledger.json',[])
        write_rows(stage/'review_queue.jsonl',[dict(record_id=r['id'],input_labels_sha256=annotation_fingerprint(r),
                   status='pending_independent_annotation') for r in rows])
        ledger=json.loads((root/'data/authored/streaming_context_v6/D3_dispositions.json').read_text())
        write_json(stage/'D3_boundary_adjudication.json',[dict(**d,candidate_ids=[r['id'] for r in rows if r['metadata'].get('v5_source_id')==d['source_id']]) for d in ledger if d['action'] in ('retain','quarantine')])
        index={r['id']:i for i,r in enumerate(rows,1)};write_json(stage/'line_index.json',index)
        for split in SPLITS:
            selected=[r for r in rows if r['metadata']['split']==split]
            write_rows(stage/'splits'/f'{split}.jsonl',selected)
            write_rows(stage/'construction_clef'/f'{split}.jsonl',[dict(state=state_text(**r['input']),questions=qs,gold=r['labels']) for r in selected])
            write_rows(stage/'clef_index'/f'{split}.jsonl',[dict(export_line=i,record_id=r['id'],records_jsonl_line=index[r['id']],
                       recommendation_withdrawn_in_target=r['metadata']['recommendation_withdrawn_in_target']) for i,r in enumerate(selected,1)])
        (stage/'annotation_standard.md').write_text((root/'docs/annotation_standard_streaming_v6.md').read_text())
        (stage/'README.md').write_text('# Context v8 full construction replacement\n\n'+str(len(rows))+' candidates across 60 product topics. Zero approved training rows.\n'
            'Retains eligible legacy snapshots and replaces the v5/v6 addition recipes with explicit context-supported designs.\n'
            'construction_clef/ is for CPU preparation and annotation review only. No approved clef/ exports exist.\n'
            'The 200 repaired v7 examples remain exposed development material. Independent annotation, token coverage and current-contract GPU smoke/reload remain pending.\n')
        contexts=list(dict.fromkeys(r['input']['customer_context'] for r in rows));context_index={c:i for i,c in enumerate(contexts)}
        view=[dict(id=r['id'],input=dict(customer_context=context_index[r['input']['customer_context']],agent_response=r['input']['agent_response']),
            labels=r['labels'],advice_present=r['advice_present'],annotation=dict(rationale=r['annotation']['rationale']),
            metadata={k:r['metadata'][k] for k in ('contrast_family','split','recommendation_withdrawn','coverage_boundary','source_refs')}) for r in rows]
        payload=dict(rows=view,contexts=contexts,questions=schema,report={**report,'authored_contrast_families':60,
                     'credit_advice_rows':sum(r['labels']['D6'] for r in rows)},dataset_id=digest({'rows':rows}))
        page=(root/'scripts/templates/streaming_review_v2.html').read_text().replace('__DATASET_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
        page=page.replace("const data=JSON.parse(document.getElementById('dataset').textContent), rows=data.rows;",
            "const data=JSON.parse(document.getElementById('dataset').textContent), rows=data.rows;\nrows.forEach(row=>{row.input.customer_context=data.contexts[row.input.customer_context];row.annotation.visible_evidence=row.input.agent_response;});")
        page=page.replace('Version 2','Context v8 · CONSTRUCTION ONLY · TRAINING BLOCKED').replace('authored trajectories','constructed trajectories')
        (stage/'review.html').write_text(page)
        old_page=(root/'data/generated/streaming_context_v7_development/review.html').read_text()
        marker='<script id="data" type="application/json">'
        before,_,tail=old_page.partition(marker);_,_,after=tail.partition('</script>')
        if not before or not after:raise ValueError('Development review template boundary missing')
        dev_page=before+marker+json.dumps(development,ensure_ascii=False).replace('<','\\u003c')+'</script>'+after
        dev_page=dev_page.replace('Context matters: natural replacement examples','Repaired v7 contrasts: 200 exposed development cases')
        (stage/'exposed_development/review.html').write_text(dev_page)
        sources=[SOURCE+'/manifest.json',FAMILIES,'data/authored/streaming_context_v8/topic_designs.py',
                 'data/authored/streaming_context_v8/development_repairs.py',EMBARGO,WITHDRAWALS,
                 'data/reviews/claude_streaming_context_v7_review/affected_ids.json',
                 'data/generated/streaming_context_v7_development/manifest.json',
                 'data/authored/streaming_context_v6/D3_dispositions.json',
                 'src/decision_guardrail/data_generation/context_v8_release.py',
                 'src/decision_guardrail/data_generation/context_v6_release.py',
                 'src/decision_guardrail/data_generation/context_v5_release.py',
                 'src/decision_guardrail/data_generation/natural_context_review.py',
                 'src/decision_guardrail/training_readiness.py',
                 'config/dataset/question_schema_streaming_v6.yaml','config/dataset/question_schema_runtime_v6.yaml',
                 'scripts/author_context_v8_pairs.py','docs/annotation_standard_streaming_v6.md']
        write_json(stage/'manifest.json',dict(schema_version=VERSION,generator=GENERATOR,records=len(rows),scenarios=len(trajectories),
            status='pre_annotation_not_trainable',training_ready=False,native_export_directory='construction_clef',annotation_ledger_path='annotation_ledger.json',
            source_hashes={s:sha(root/s) for s in sources},files={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}))
        stage.rename(output)
    return report


def verify(output):
    manifest=verify_files(output)
    if manifest['generator']!=GENERATOR or manifest['schema_version']!=VERSION or manifest['training_ready'] is not False:
        raise ValueError('Wrong construction replacement contract')
    if manifest['native_export_directory']!='construction_clef' or (output/'clef').exists():raise ValueError('Unapproved native training exports')
    rows=read_rows(output/'records.jsonl');exposed=read_rows(output/'exposed_development/records.jsonl')
    qs=json.loads((output/'native_questions.json').read_text())
    report,_=quality(rows,exposed,qs);saved=json.loads((output/'quality_report.json').read_text())
    for key,value in report.items():
        if saved[key]!=value:raise ValueError('Quality report mismatch: '+key)
    if manifest['records']!=len(rows):raise ValueError('Manifest record count differs')
    withdrawn={r['record_id'] for r in read_rows(output/'withdrawal_ledger.jsonl')}
    embargo=set(json.loads((output/'exposure_embargo.json').read_text())['record_ids'])
    source_ids={r['metadata'].get('v6_source_id') for r in rows if r['metadata'].get('v6_source_id')}
    if len(withdrawn)!=3097 or len(embargo)!=121 or source_ids & (withdrawn|embargo):
        raise ValueError('Withdrawn or embargoed source was promoted')
    source_dispositions=read_rows(output/'source_dispositions.jsonl')
    if len(source_dispositions)!=19838 or len({d['source_id'] for d in source_dispositions})!=19838:
        raise ValueError('Source dispositions are incomplete')
    kept={d['source_id']:d['candidate_id'] for d in source_dispositions if d['disposition']=='retain'}
    if set(kept)!=source_ids or set(kept.values())!={r['id'] for r in rows if r['metadata']['origin']=='retained_v4'}:
        raise ValueError('Retained source ledger differs from candidates')
    by_id={r['id']:r for r in rows};support=read_rows(output/'context_support_ledger.jsonl')
    if len({(r['family'],r['point']) for r in support})!=len(support):raise ValueError('Repeated support-ledger cell')
    for point in support:
        row=by_id[point['record_id']];ctx=row['input']['customer_context']
        if row['metadata']['origin']!='v8_natural_context_design':raise ValueError('Support ledger refers to a legacy record')
        if set(point['required_context_tags'])!={s['tag'] for s in point['context_support_spans']}:raise ValueError('Missing required support')
        for span in point['context_support_spans']:
            if ctx[span['start']:span['end']]!=span['text']:raise ValueError('Context support span differs')
    for split in SPLITS:
        selected=[r for r in rows if r['metadata']['split']==split]
        if read_rows(output/'construction_clef'/f'{split}.jsonl')!=[dict(state=state_text(**r['input']),questions=qs,gold=r['labels']) for r in selected]:
            raise ValueError('Native construction export differs')
    if json.loads((output/'annotation_ledger.json').read_text())!=[]:raise ValueError('Construction release has a nonempty approval ledger')
    return saved
