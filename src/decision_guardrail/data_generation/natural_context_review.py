"""Development-only natural replacement designs; never a training release.

Full v6 generation remains frozen for reproducibility. Its synthetic echo-based
counterexamples are withdrawn from future promotion, not silently relabelled.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import tempfile
from .context_v5_review import digest,sha,write_json,write_rows
from ..schema import decision_violations

CLASSES=('factual_information','general_advice','personal_advice','other')
ROLES={'fact':'factual_information','general':'general_advice','personal':'personal_advice','other':'other'}


def reply_body(text):
    """Ignore acknowledgement/quotation wrappers when auditing same-context repeats.

    Deliberately audit this alongside exact input matching; it is not a semantic
    duplicate detector and must not be used to auto-label or auto-approve rows.
    """
    patterns=(
        r'^Your statement was ["“].*?["”]\.\s*',
        r'^I acknowledge your statement, ["“].*?["”]\.\s*',
        r'^I have recorded your statement as ["“].*?["”]; that records the request rather than assessing it\.\s*',
        r'^The note of your statement reads ["“].*?["”]\. Recording it does not determine a selection\.\s*',
        r'^The words in your statement are ["“].*?["”]\. Acknowledgement alone is not a product assessment\.\s*',
    )
    for pattern in patterns:text=re.sub(pattern,'',text,flags=re.I|re.S)
    return ' '.join(re.findall(r"[a-z0-9']+",text.casefold()))


def duplicate_bodies(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['input']['customer_context'],reply_body(r['input']['agent_response']))].append(r['id'])
    return [ids for ids in groups.values() if len(ids)>1]


def full_turn_echoes(rows):
    found=[]
    for r in rows:
        target=r['input']['agent_response'].casefold()
        turns=[re.sub(r'^(CUSTOMER|AGENT|ASSISTANT):\s*','',line).strip() for line in r['input']['customer_context'].replace('\\n','\n').splitlines() if line.startswith('CUSTOMER:')]
        if any(len(t.split())>=8 and t.casefold() in target for t in turns):found.append(r['id'])
    return found


def construct(families):
    records=[]
    for family in families:
        contexts={'generic': 'CUSTOMER: '+family['generic'],
                  'disclosed_need':'CUSTOMER: '+family['need'],
                  'explicit_generic':'CUSTOMER: '+family['explicit_generic'],
                  'multi_turn':'\n'.join(family['history'])}
        for context_kind,context in contexts.items():
            for speech in ('fact','general','personal','other','implicit'):
                kind=ROLES.get(speech,'personal_advice' if context_kind in ('disclosed_need','multi_turn') else 'general_advice')
                advice=kind in ('general_advice','personal_advice')
                actual_basis=speech=='personal' and context_kind!='generic' and family['personal_D3']
                labels=dict(D1=True,D2=advice,D3=bool(actual_basis),D4=kind=='personal_advice',
                            D5=kind=='factual_information',D6=advice and family['credit'],D7=kind)
                if decision_violations(labels):raise ValueError('Inconsistent authored design')
                inp=dict(customer_context=context,agent_response=family[speech])
                rationale={
                    'fact':'Describes product operation without endorsing a selection, including when the customer discloses a need.',
                    'general':'Expresses a product-feature opinion without making an individual selection or using this customer as its basis.',
                    'other':'Asks a clarification question; no substantive product fact or endorsed recommendation has yet been expressed.',
                    'personal':('Makes an individual selection using disclosed circumstances. An explicit request for general information does not override the response actually given.' if actual_basis else
                                'Asserts an individual selection without a disclosed basis. Unsupported or unsolicited tailoring still asserts suitability; D3 remains false.'),
                    'implicit':('The selection answers the customer’s request to choose for the disclosed need; suitability is implied across turns. D3 remains false because the response does not demonstrate a basis beyond that implication.' if kind=='personal_advice' else
                                'A general request receives an unqualified product opinion without individual suitability wording; no demonstrable customer basis is used.'),
                }[speech]
                records.append(dict(id='natural-dev-'+digest(inp)[:24],input=inp,labels=labels,
                    annotation=dict(status='pending_independent_annotation',independent_review=False,rationale=rationale,
                                    provenance='Codex-authored development design, not blind validation'),
                    metadata=dict(family=family['id'],context_kind=context_kind,speech_act=speech,split=None,
                                  native_export_eligible=False,exposed_development_only=True)))
    if len({digest(r['input']) for r in records})!=len(records):raise ValueError('Duplicate full input')
    if duplicate_bodies(records):raise ValueError('Same-context reply body duplicated')
    if full_turn_echoes(records):raise ValueError('Full customer turn copied into replacement reply')
    return records


def build(root,output):
    if output.exists():raise ValueError('Choose a new output path')
    authored=root/'data/authored/streaming_context_v7/natural_families.json'
    records=construct(json.loads(authored.read_text()))
    source=root/'data/generated/streaming_context_v6_release_r3'
    old=[json.loads(l) for l in (source/'records.jsonl').read_text().splitlines()]
    withdrawn=[r for r in old if r['metadata']['origin']=='v6_crossed_response_design']
    reviewed=[json.loads(l) for l in (root/'data/reviews/claude_streaming_context_v6_review/record_review.jsonl').read_text().splitlines()]
    sampled={r.get('record_id',r.get('id')) for r in reviewed}
    # Any future release must exclude the complete linked units, not just the
    # 80 sampled rows. Save this embargo now without changing frozen v6.
    units={u for r in old if r['id'] in sampled for u in r['metadata']['units']}
    while True:
        linked={u for r in old if units.intersection(r['metadata']['units']) for u in r['metadata']['units']}
        if linked<=units:break
        units|=linked
    exposed=[r['id'] for r in old if units.intersection(r['metadata']['units'])]
    report=dict(status='development_replacement_batch_not_a_dataset_release',records=len(records),families=len({r['metadata']['family'] for r in records}),
                training_ready=False,approved_training_records=0,full_turn_echoes=0,same_context_duplicate_bodies=0,
                class_counts=dict(Counter(r['labels']['D7'] for r in records)),
                withdrawn_v6_counterexamples=len(withdrawn),v6_review_exposed_linked_records=len(exposed),
                limitations=['Authored and exposed development examples; not an independent challenge set.',
                            'This batch does not replace all 60 topic families or certify the retained legacy corpus.',
                            'No split assignments or native training exports are supplied.',
                            'Independent annotation, native token coverage and an approved GPU smoke/reload remain pending.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='natural-review-') as tmp:
        stage=Path(tmp)/'batch';stage.mkdir()
        write_rows(stage/'records.jsonl',records);write_json(stage/'quality_report.json',report)
        write_rows(stage/'withdrawn_v6_counterexamples.jsonl',[dict(record_id=r['id'],reason='withdrawn_echo_template_design',future_promotion_allowed=False) for r in withdrawn])
        write_json(stage/'exposed_v6_review_units.json',dict(record_ids=exposed,units=sorted(units),future_candidate_eligible=False))
        write_json(stage/'annotation_ledger.json',[])
        payload=json.dumps(records,ensure_ascii=False).replace('<','\\u003c')
        page='''<!doctype html><meta charset="utf-8"><title>Natural context replacement review</title>
<style>body{font:17px system-ui;background:#f4f7f7;color:#16313b;max-width:1100px;margin:40px auto}article{background:white;border:1px solid #d7e2e5;border-radius:14px;padding:24px;margin:20px 0}pre{white-space:pre-wrap;font:inherit}select{font:inherit;padding:8px}small{color:#526974}.banner{padding:20px;background:#e2f1ef;border-radius:12px}</style>
<h1>Context matters: natural replacement examples</h1><p class="banner">Exposed development batch · No training approval · No held-out scores</p><p>Choose a family to compare the same responses across generic requests, disclosed needs and multiple turns. Labels are proposed and need independent annotation.</p><select id="family"></select><main id="rows"></main><script id="data" type="application/json">'''+payload+'''</script><script>
const records=JSON.parse(document.getElementById('data').textContent),select=document.getElementById('family'),container=document.getElementById('rows');
for(const name of [...new Set(records.map(r=>r.metadata.family))]){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');select.append(o)}
function show(){container.replaceChildren();for(const r of records.filter(r=>r.metadata.family===select.value)){const a=document.createElement('article');for(const [tag,text] of [['small',r.metadata.context_kind+' · '+r.metadata.speech_act],['pre',r.input.customer_context],['p','RESPONSE: '+r.input.agent_response],['strong',r.labels.D7.replaceAll('_',' ')],['p',Object.entries(r.labels).filter(([k])=>k!=='D7').map(([k,v])=>k+': '+(v?'Yes':'No')).join(' · ')],['small',r.annotation.rationale]]){const e=document.createElement(tag);e.textContent=text;a.append(e)}container.append(a)}}select.addEventListener('change',show);show();</script>'''
        (stage/'review.html').write_text(page)
        (stage/'README.md').write_text('# Natural context development replacements\n\n200 proposed examples across 10 families. All examples are exposed development only, split=null, with zero approved training rows. This is not a replacement full dataset release.\n\nSee quality_report.json and the v6 withdrawal/exposure ledgers.\n')
        write_json(stage/'manifest.json',dict(status=report['status'],training_ready=False,records=len(records),
                    source_hashes={str(p.relative_to(root)):sha(p) for p in [authored,source/'manifest.json',Path(__file__).resolve()]},
                    files={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}))
        stage.rename(output)
    return report
