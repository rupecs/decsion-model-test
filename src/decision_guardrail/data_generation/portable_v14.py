"""Portable v14 assembly. Only the historical input boundary differs from the original; see docs/reproduction.md."""
from copy import deepcopy
import hashlib
from collections import Counter
import json
from pathlib import Path

from .context_v12_authoring import labels, render
from .context_v5_release import merge_rows
from .context_v5_review import digest, write_json, write_rows
from .context_v6_release import VERSION
from .dialogue_exposure import design_fingerprint, design_ids, exposure_hits, read_jsonl, sha
from .dialogue_exposure_policy import load_policy

INHERITED = 'outputs/context_v14_corrections_20261025/stage02/unexposed_working_pool_not_release.jsonl'
BASE = 'data/generated/streaming_context_v13_release'
POLICY = 'config/dataset/dialogue_exposure_policy_v14.json'


def inherited(root):
    """Load only records already present in v14, not any older release.

    Their original inherited identities and row provenance are preserved. This
    portable boundary does not claim to rerun unavailable historical reviews.
    """
    import gzip
    with gzip.open(root/'data/authored/streaming_context_v14/retained_v14_members.json.gz','rt',encoding='utf-8') as f:
        seed=json.load(f)
    return seed['records'],seed['designs'],seed['bindings'],seed['dispositions']


def assemble(root:Path):
    rows,designs,bindings,dispositions=inherited(root);raw=[]
    source=json.loads((root/'data/authored/streaming_context_v14/products.json').read_text())
    retained_design_count=len(designs)
    services=json.loads((root/'data/authored/streaming_context_v14/services.json').read_text())
    def add(d,history,path,text,profile,stages=None,mixed=False,scope=True,basis=''):
        context,asr,paired=render(history);key=hashlib.sha256(context.encode()).hexdigest()
        if key in bindings and bindings[key]!=paired:raise ValueError('Conflicting exact dialogue binding')
        bindings[key]=paired;gold=labels(profile,d['domain']=='credit',scope)
        spans=[]
        if gold['D3']:
            # The entire opening is an actual disclosed financial need, never an operational question.
            opening=context.splitlines()[0].removeprefix('CUSTOMER: ')
            spans=[dict(tag='actual_need',start=10,end=10+len(opening),text=opening)]
        inp=dict(customer_context=context,agent_response=text)
        index,total=stages or (0,1)
        ref=dict(scenario_id=d['design_id']+':'+digest(context)[:16]+':'+path,checkpoint_index=index,
                 total_checkpoints=total,is_final=index==total-1,value_added='Explicit audible stage or response alternative')
        raw.append(dict(id='sctxv14-'+digest(inp)[:24],schema_version=VERSION,input=inp,labels=gold,advice_present=gold['D2'],
            annotation=dict(profile=profile,status='pending_independent_annotation',independent_review=False,
                author='Codex authored proposals',rule_version='streaming-context-dialogue-v14',
                rationale='Proposed speech-act labels. Procedures are not endorsements; actual-basis use and individual suitability are assessed separately.',
                visible_evidence=text,basis_relevance=None,conversation_coherence=None,
                evidence=dict(context_support_spans=spans,target_spans=[dict(start=0,end=len(text),text=text)])),
            metadata=dict(parent_topic=d['product_topic'],contrast_family=d['design_id'],design_sha256=design_fingerprint(d),
                origin='v14_fresh_authored_dialogue',split=d['split'],split_group=d['product_topic'],units=['v14-case:'+d['design_id']],
                source_ids=[],source_refs=[ref],scenario_product_domains=[d['domain']],call_category=d['call_category'],
                customer_situation_id='ctx-'+digest(context)[:20],customer_disclosure=bool(basis),
                customer_basis_type='actual_disclosure' if basis else 'operational_information_request',
                construction_recipe='dialogue_'+str(len(history)),coverage_boundary=path,voice_style=True,asr_imperfections=asr,
                mixed_content=mixed,disclaimer=False,intentional_speech_act_drift=False,recommendation_withdrawn_in_target=False,
                recommendation_withdrawn=False,boundary_adjudication_required=True,
                context_kind='actual_or_operational' if basis else 'unsolicited_choice_control',
                lineage=[dict(source_design_id=d['design_id'],authored_case=d['design_id'],authored_point=path)])))
    def path(d,h,name,texts,profiles,scope=True,mixed=False,basis=''):
        for i,(text,profile) in enumerate(zip(texts,profiles,strict=True)):
            if i and not text.startswith(texts[i-1]):raise ValueError('Non-extending authored voice path: '+d['design_id']+name)
            add(d,h,'audible_stages:'+name,text,profile,(i,len(texts)),mixed and profile not in ('fact','other'),scope,basis)
    for p in source['PRODUCTS']:
        for case,goal in enumerate((*p['goals'],*p['operations'])):
            actual=case<len(p['goals']);profile=('linked' if case==1 else 'basis_only') if actual else 'implicit'
            probe,reply=(p['probe'],p['answer']) if actual else (p['op_probe'],p['op_answer'])
            hist=[('CUSTOMER',goal),('AGENT',probe),('CUSTOMER',reply),('AGENT',p['explanation']),
                  ('CUSTOMER',p['followup']),('AGENT',p['followup_answer']),('CUSTOMER',p['closing'])]
            d=dict(design_id='v14-'+p['key']+'-'+str(case),product_topic='v14-'+p['key'],domain=p['domain'],
                call_category=p['call'],customer_goal=goal,decision_basis=goal if actual else 'Unsolicited individual suitability with no disclosed substantive financial basis: '+goal,
                split=p['split'],source_design_ids=[],proposed_profile=profile)
            designs.append(d)
            subject,predicate=p['fact'];fact=subject+' '+predicate
            if actual:
                personal_prefix,personal=p['basis'] if profile=='basis_only' else p['choice']
            else:
                personal=source['UNSUPPORTED'][p['key']]
                personal_prefix=personal.split(' is ',1)[0] if ' is ' in personal else 'I recommend'
                if not personal.startswith(personal_prefix):raise ValueError('Missing operational voice boundary')
            gen_prefix,general=p['general'];q=p['question'];partial_q=' '.join(q.split()[:3])
            for length in (1,3,5,7):
                h=hist[:4]+[('CUSTOMER',p['closing'])] if length==5 else hist[:length];basis=goal if actual else ''
                path(d,h,'fact',[subject,fact],['other','fact'],basis=basis)
                add(d,h,'comparison_procedure',p['procedure'],'fact',basis=basis)
                add(d,h,'second_fact',p['extra_fact'],'fact',basis=basis)
                path(d,h,'general',[gen_prefix,general],['other','general'],basis=basis)
                path(d,h,'fact_then_general',[fact+' '+gen_prefix,fact+' '+general],['fact','general'],mixed=True,basis=basis)
                if personal_prefix:
                    path(d,h,'personal',[personal_prefix,personal],['other',profile],basis=basis)
                else:add(d,h,'personal_complete',personal,profile,basis=basis)
                if actual:
                    path(d,h,'personal_question',[personal+' '+partial_q,personal+' '+q],[profile,profile],basis=basis)
                    path(d,h,'mixed',[fact+' '+personal_prefix,fact+' '+personal+' '+partial_q,fact+' '+personal+' '+q],
                         ['fact',profile,profile],mixed=True,basis=basis)
                    add(d,h,'mixed_complete',fact+' '+personal,profile,mixed=True,basis=basis)
                else:
                    path(d,h,'mixed',[fact+' '+personal_prefix,fact+' '+personal],['fact',profile],mixed=True)
                    add(d,h,'procedure_then_unsupported_choice',p['procedure']+' '+personal,profile,mixed=True)
                    add(d,h,'categorical_opinion',source['CATEGORY_OPINIONS'][p['key']],'general')
                    path(d,h,'fact_then_categorical_opinion',[fact,fact+' '+source['CATEGORY_OPINIONS'][p['key']]],['fact','general'],mixed=True)
                add(d,h,'standalone_followup_offer',q,'other',basis=basis)
                for oi,other in enumerate(p['other']):add(d,h,'other_complete_'+str(oi),other,'other',basis=basis,scope=p['other_scopes'][oi])
    for item in services:
        key,split,call,scope,goal,probe,reply,explain,follow,explanation,closing,factparts,procedure,other1,other2=item
        hist=[('CUSTOMER',goal),('AGENT',probe),('CUSTOMER',reply),('AGENT',explain),('CUSTOMER',follow),('AGENT',explanation),('CUSTOMER',closing)]
        d=dict(design_id='v14-service-'+key,product_topic='v14-service-'+key,domain='service',call_category=call,
               customer_goal=goal,decision_basis='Operational question: '+goal,split=split,source_design_ids=[],proposed_profile='fact_other_controls')
        designs.append(d);subject,predicate=factparts;fact=subject+' '+predicate
        for length in (1,3,5,7):
            h=hist[:4]+[('CUSTOMER',closing)] if length==5 else hist[:length]
            path(d,h,'fact',[subject,fact],['other','fact'],scope=scope)
            add(d,h,'procedure',procedure,'fact',scope=scope)
            add(d,h,'clarification',other1,'other',scope=scope)
            add(d,h,'pause',other2,'other',scope=scope)
    fresh, merged=merge_rows(raw)
    # Do not publish ambiguous generic fragments that duplicate a target in a different partition.
    by_target={}
    for r in [*rows,*fresh]:by_target.setdefault(r['input']['agent_response'],set()).add(r['metadata']['split'])
    conflicts={t for t,splits in by_target.items() if len(splits)>1}
    removed=[];kept=[]
    for r in fresh:
        if r['input']['agent_response'] in conflicts:
            if any(ref['is_final'] for ref in r['metadata']['source_refs']) and r['labels']['D7']!='other':raise ValueError('Complete substantive target crosses partitions: '+r['input']['agent_response'])
            removed.append(dict(record_id=r['id'],reason='shared_routine_other_or_unfinished_fragment_excluded_without_rewording'))
        else:kept.append(r)
    # Reindex filtered paths, preserving their order and labels.
    refs={}
    for r in kept:
        for ref in r['metadata']['source_refs']:refs.setdefault(ref['scenario_id'],[]).append(ref)
    for path_refs in refs.values():
        for i,ref in enumerate(sorted(path_refs,key=lambda x:x['checkpoint_index'])):
            ref.update(checkpoint_index=i,total_checkpoints=len(path_refs),is_final=i==len(path_refs)-1)
    result=rows+kept
    report=dict(records=len(result),retained_v13=len(rows),fresh=len(kept),retained_designs=retained_design_count,fresh_designs=len(designs)-retained_design_count,
        category_history_exclusions=sum(d['reason']=='unresolved_category_actual_application_history' for d in dispositions),
        inherited_quarantines_by_reason=dict(Counter(d['reason'] for d in dispositions)),short_fragment_exclusions=removed,training_ready=False,
        approved_training_records=0,scope='candidate_pack_not_published_release')
    return result,designs,bindings,dict(report=report,dispositions=dispositions,
        retention=[dict(record_id=r['id'],source=BASE+'/records.jsonl',source_manifest_sha256='bf316a2f2c897fac989fe6f94858d78058e6bf4ef9543059108bfa6f19653d10',
                        source_row_sha256=digest(r),split=r['metadata']['split']) for r in rows])


def write_pack(root:Path,output:Path):
    if output.exists() or not output.resolve().is_relative_to(root.resolve()):raise ValueError('Use a new in-project authoring path')
    rows,designs,bindings,report=assemble(root)
    output.mkdir(parents=True)
    write_rows(output/'records.jsonl',rows);write_rows(output/'designs.jsonl',designs)
    write_json(output/'dialogue_bindings.json',bindings);write_json(output/'assembly_report.json',report)
    return report['report']
