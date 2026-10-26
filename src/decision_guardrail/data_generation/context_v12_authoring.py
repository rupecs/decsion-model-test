"""Assemble the explicitly authored v12 proposals, preserving exact dialogue bindings.

This module writes an authoring pack, not a published or approved dataset. There
is no quota loop, model/teacher call, invented speaker or surface greeting padding.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

from .context_v5_release import merge_rows
from .context_v5_review import digest, write_json, write_rows
from .context_v6_release import VERSION
from .context_v10_content import asr_text
from .dialogue_exposure import design_fingerprint


def labels(profile: str, credit: bool, scope: bool) -> dict:
    d2,d3,d4,d5={"fact":(False,False,False,True),"other":(False,False,False,False),
        "general":(True,False,False,False),"implicit":(True,False,True,False),
        "linked":(True,True,True,False),"basis_only":(True,True,False,False)}[profile]
    return dict(D1=scope,D2=d2,D3=d3,D4=d4,D5=d5,D6=credit and d2,
                D7=("personal_advice" if d3 or d4 else "general_advice") if d2 else "factual_information" if d5 else "other")


def render(turns: list[tuple[str,str]]) -> tuple[str,bool,list[dict]]:
    plain="\n".join(role+": "+text for role,text in turns)
    # Replace a transcript surface, never count it as an extra conversation.
    asr=len(turns)>1 and int(digest(plain)[:8],16)%5==0
    final=[(role,asr_text(text) if asr else text) for role,text in turns]
    text="\n".join(role+": "+value for role,value in final)
    bindings=[dict(answer_turn_index=i,customer_turn=final[i-1][1],agent_turn=value)
              for i,(role,value) in enumerate(final) if role=="AGENT"]
    return text,asr,bindings


def assemble(root: Path) -> tuple[list[dict],list[dict],dict,dict]:
    source=root/"data/authored/streaming_context_v12"
    products=runpy.run_path(str(source/"product_designs.py"))["PRODUCTS"]
    dialogues=runpy.run_path(str(source/"product_dialogues.py"))["DIALOGUES"]
    services=runpy.run_path(str(source/"service_designs.py"))["SERVICES"]
    implicit=runpy.run_path(str(source/"implicit_selections.py"))["IMPLICIT"]
    generic_questions=runpy.run_path(str(source/"generic_questions.py"))["QUESTIONS"]
    generic_recommendations=runpy.run_path(str(source/"generic_recommendations.py"))["RECOMMENDATIONS"]
    case_responses=runpy.run_path(str(source/"case_responses.py"))
    prior_controls=runpy.run_path(str(source/"prior_advice_controls.py"))["CONTROLS"]
    plan=json.loads((source/"split_plan.json").read_text())
    topic_plan={x["topic"]:x for x in plan["product_topics"]}
    if set(products)!=set(dialogues) or set(products)!=set(topic_plan) or set(products)!=set(generic_questions):
        raise ValueError("Incomplete explicit product or dialogue coverage")
    raw,designs,bindings=[],[],{}

    def add(design,ctx,kind,text,profile,actual,need,mixed=False,scope=True,prefix=None):
        context,asr,paired=render(ctx)
        key=hashlib.sha256(context.encode()).hexdigest()
        if key in bindings and bindings[key]!=paired:raise ValueError("Inconsistent dialogue binding")
        bindings[key]=paired
        gold=labels(profile,design["domain"]=="credit",scope)
        spans=[]
        if gold["D3"]:
            evidence=asr_text(need) if asr else need
            start=context.find(evidence)
            if not actual or start<0:raise ValueError("Missing actual basis in this exact history")
            spans=[dict(tag="actual_need",start=start,end=start+len(evidence),text=evidence)]
        inp=dict(customer_context=context,agent_response=text)
        scenario=design["design_id"]+":"+ctx_id(ctx)+":"+(prefix["path"] if prefix else kind)
        raw.append(dict(id="sctxv12-"+digest(inp)[:24],schema_version=VERSION,input=inp,labels=gold,advice_present=gold["D2"],
            annotation=dict(profile=profile,status="pending_independent_annotation",independent_review=False,
                author="Codex construction",rule_version="streaming-context-dialogue-v12",
                rationale="Classify only the available target in this exact conversation. Proposed product-use and suitability boundaries require independent adjudication.",
                visible_evidence=text,basis_relevance=None,conversation_coherence=None,
                evidence=dict(context_support_spans=spans,target_spans=[dict(start=0,end=len(text),text=text)])),
            metadata=dict(parent_topic=design["product_topic"],contrast_family=design["design_id"],
                design_sha256=design_fingerprint(design),origin="v12_fresh_design",split=design["split"],
                split_group=design["product_topic"],units=["v12-case:"+design["design_id"]],source_ids=[],
                source_refs=[dict(scenario_id=scenario,checkpoint_index=prefix["index"] if prefix else 0,
                    total_checkpoints=prefix["total"] if prefix else 1,is_final=prefix["final"] if prefix else True,
                    value_added="Authored speech-act alternative or explicit audible stage")],
                scenario_product_domains=[design["domain"]],call_category=design["call_category"],
                customer_situation_id="ctx-"+digest(context)[:20],customer_disclosure=actual,
                customer_basis_type="actual_disclosure" if actual else "generic_or_operational",
                construction_recipe="dialogue_"+str(len(ctx)),coverage_boundary=kind,
                voice_style=True,asr_imperfections=asr,mixed_content=mixed,disclaimer=False,
                intentional_speech_act_drift=False,recommendation_withdrawn_in_target=False,recommendation_withdrawn=False,
                boundary_adjudication_required=profile in ("linked","implicit","basis_only") or (actual and profile=="general"),
                difficulty="complex" if len(ctx)>=7 else "boundary",recommendation_style=profile,
                synthetic=True,independent_annotation=False,native_export_eligible=True,
                lineage=[dict(version="streaming-context-dialogue-v12",authored_case=design["design_id"],authored_point=kind)])))

    for topic,p in products.items():
        info=topic_plan[topic];dialogue=dialogues[topic]
        generic=[("CUSTOMER",p["generic_request"])]
        generic_short=generic+[("AGENT",p["generic_question"]),("CUSTOMER",p["generic_focus"])]
        for index,case in enumerate(p["cases"]):
            goal,choice_segments,reason,closing,q1,q2=case
            subject,predicate=choice_segments.split("|");choice=subject+" "+predicate
            interview,reply,explanation=dialogue["interviews"][index]
            case_key=topic+"-"+"ab"[index]
            followup,followup_answer=case_responses["FOLLOWUPS"].get(case_key,(dialogue["followup"],dialogue["answer"]))
            initial=[("CUSTOMER",goal)]
            short=initial+[("AGENT",interview),("CUSTOMER",reply)]
            middle=short+[("AGENT",explanation),("CUSTOMER",closing)]
            long=short+[("AGENT",explanation),("CUSTOMER",followup),("AGENT",followup_answer),("CUSTOMER",closing)]
            design=dict(design_id=f"v12-{topic}-{'ab'[index]}",product_topic=topic,domain=info["domain"],
                call_category=info["call"],customer_goal=goal,decision_basis=reason,split=info["split"],source_design_ids=[])
            designs.append(design)
            for history,actual in ((initial,True),(short,True),(middle,True),(long,True),(generic,False),(generic_short,False)):
                response=case_responses["RESPONSES"].get(case_key,p) if actual else p
                fact_subject,fact_tail=response["fact_segments"];fact=fact_subject+" "+fact_tail
                condition,opinion=response["condition"],response["opinion"]
                profile="linked" if actual else "general"
                ending=reason if actual else q1
                target_q1,target_q2=(q1,q2) if actual else generic_questions[topic]
                candidates=[("fact",fact,"fact",False),("fact_conditions",fact+" "+condition,"fact",False),
                    ("fact_question",condition+" "+target_q1,"fact",False),
                    ("opinion",opinion,"general",False),("fact_opinion",fact+" "+opinion,"general",True),
                    ("clarification",target_q1,"other",False),("alternative_question",target_q2,"other",False)]
                if actual:
                    candidates.extend([("selection",choice,profile,False),("selection_basis",choice+" "+ending,profile,False),
                        ("mixed_selection",fact+" "+choice,profile,True),("mixed_basis",fact+" "+choice+" "+ending,profile,True)])
                else:
                    conditional=generic_recommendations[topic]
                    candidates.extend([("opinion_question",conditional+" "+target_q1,"general",False),
                        ("mixed_opinion_question",fact+" "+conditional+" "+target_q1,"general",True)])
                if actual and topic in implicit:
                    candidates.append(("implicit_selection",implicit[topic][index],"implicit",False))
                    if index==0:candidates.append(("basis_without_suitability",reason+" "+opinion,"basis_only",False))
                for kind,text,label,mixed in candidates:add(design,history,kind,text,label,actual,goal,mixed)
                paths=[("facts",[fact_subject,fact,fact+" "+condition],["other","fact","fact"],[False]*3)]
                if actual:
                    # Observe a genuine unfinished reason after an endorsement.
                    # The already audible advice remains present. This prevents
                    # nonterminal/unfinished syntax becoming a no-advice cue.
                    reason_prefix=" ".join(ending.split()[:3])
                    info_texts=[fact,fact+" "+subject,fact+" "+choice]
                    info_profiles=["fact","fact",profile];info_mixed=[False,False,True]
                    direct_texts=[subject,choice];direct_profiles=["other",profile]
                    if len(history) in (1,7):
                        info_texts.append(fact+" "+choice+" "+reason_prefix)
                        info_profiles.append(profile);info_mixed.append(True)
                    else:
                        direct_texts.append(choice+" "+reason_prefix);direct_profiles.append(profile)
                    info_texts.append(fact+" "+choice+" "+ending);info_profiles.append(profile);info_mixed.append(True)
                    direct_texts.append(choice+" "+ending);direct_profiles.append(profile)
                    paths.extend([("advice",info_texts,info_profiles,info_mixed),
                                  ("direct",direct_texts,direct_profiles,[False]*len(direct_texts))])
                else:
                    # A generic product opinion also stays advice while a
                    # following clarification question is only partly audible.
                    partial_q=" ".join(target_q1.split()[:2])
                    paths.append(("generic_opinion",[conditional,conditional+" "+partial_q,conditional+" "+target_q1],
                                  ["general"]*3,[False]*3))
                for path,texts,profiles,mixeds in paths:
                    for i,(text,label,mixed) in enumerate(zip(texts,profiles,mixeds,strict=True)):
                        add(design,history,"audible_"+path+"_"+str(i),text,label,actual,goal,mixed,
                            prefix=dict(path="audible_stages:"+path,index=i,total=len(texts),final=i==len(texts)-1))
            if case_key in prior_controls:
                question,segments,continuation=prior_controls[case_key]
                first,tail=segments.split("|")
                history=[("CUSTOMER",goal),("AGENT",choice),("CUSTOMER",question)]
                texts=[first,first+" "+tail,first+" "+tail+" "+continuation]
                for i,text in enumerate(texts):
                    add(design,history,"prior_advice_fact_"+str(i),text,"other" if i==0 else "fact",True,goal,
                        prefix=dict(path="audible_stages:prior_turn_advice",index=i,total=3,final=i==2))

    expected={(c,s) for c,ss in plan["service_pairs_needed"].items() for s in ss}
    if {(s["call"],s["split"]) for s in services}!=expected:raise ValueError("Missing planned service pair")
    for s in services:
        fact_subject,fact_tail=s["fact_segments"];fact=fact_subject+" "+fact_tail
        for index,case in enumerate(s["cases"]):
            opening,answer,explanation,closing=case
            initial=[("CUSTOMER",opening)]
            short=initial+[("AGENT",s["question"]),("CUSTOMER",answer)]
            long=short+[("AGENT",explanation),("CUSTOMER",s["followup"]),("AGENT",s["answer"]),("CUSTOMER",closing)]
            design=dict(design_id=f"v12-service-{s['key']}-{'ab'[index]}",product_topic="v12-service-"+s["key"],domain="service",
                call_category=s["call"],customer_goal=opening,decision_basis=explanation,split=s["split"],source_design_ids=[])
            designs.append(design)
            for history in (initial,short,long):
                candidates=[("fact",fact,"fact"),("conditions",fact+" "+s["condition"],"fact"),
                    ("fact_question",s["condition"]+" "+s["nextq"],"fact"),("procedure",s["procedure"],"fact"),
                    ("procedure_question",s["procedure"]+" "+s["otherq"],"fact"),
                    ("clarification",s["nextq"],"other"),("alternative_question",s["otherq"],"other")]
                for kind,text,profile in candidates:add(design,history,kind,text,profile,True,opening,scope=s["scope"])
                for i,text in enumerate((fact_subject,fact,fact+" "+s["condition"])):
                    add(design,history,"audible_fact_"+str(i),text,"other" if i==0 else "fact",True,opening,scope=s["scope"],
                        prefix=dict(path="audible_stages:service_fact",index=i,total=3,final=i==2))
    rows,merged=merge_rows(raw)
    # Identical inputs may legitimately close more than one authored path.
    return rows,designs,bindings,dict(raw_records=len(raw),records=len(rows),merged=merged,designs=len(designs))


def ctx_id(turns):return digest(turns)[:16]


def write_pack(root: Path, output: Path) -> dict:
    if output.exists():raise ValueError("Use a new authoring-pack path")
    rows,designs,bindings,report=assemble(root)
    output.mkdir(parents=True)
    write_rows(output/"records.jsonl",rows);write_rows(output/"designs.jsonl",designs)
    write_json(output/"dialogue_bindings.json",bindings);write_json(output/"assembly_report.json",report)
    return report
