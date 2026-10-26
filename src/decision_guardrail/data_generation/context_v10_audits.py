"""Comparable lexical diagnostics and coverage tables; no model-quality claims."""
from collections import Counter,defaultdict
import csv
import math
import re


def tokens(text):
    return re.findall(r"[a-z0-9']+",text.casefold())


def low_support_cues(rows,support=30,precision=.95):
    counts=defaultdict(Counter)
    for r in rows:
        words=tokens(r['input']['agent_response'])
        for gram in {tuple(words[i:i+k]) for k in range(1,6) for i in range(len(words)-k+1)}:
            counts[gram][r['labels']['D7']]+=1
    result=[]
    for gram,counts_ in counts.items():
        n=sum(counts_.values());label,k=counts_.most_common(1)[0]
        if n>=support and k/n>precision:
            result.append(dict(phrase=' '.join(gram),support=n,dominant_class=label,precision=k/n))
    return sorted(result,key=lambda x:(-x['support'],x['phrase']))


def word_baseline(rows,field):
    """Same unigram+bigram multinomial NB family as the v9 reviewer diagnostic."""
    def feats(r):
        words=tokens(r['input'][field])
        return words+['_'.join(p) for p in zip(words,words[1:])]
    classes=Counter();words=defaultdict(Counter);vocabulary=set()
    for r in rows:
        if r['metadata']['split']!='train':continue
        y=r['labels']['D7'];f=feats(r);classes[y]+=1;words[y].update(f);vocabulary.update(f)
    totals={y:sum(c.values()) for y,c in words.items()};correct=0;test=0
    for r in rows:
        if r['metadata']['split']!='test':continue
        f=feats(r)
        winner=max(classes,key=lambda y:math.log(classes[y])+sum(math.log((words[y][w]+1)/(totals[y]+len(vocabulary))) for w in f))
        test+=1;correct+=winner==r['labels']['D7']
    return dict(correct=correct,test_rows=test,accuracy=correct/test,
                interpretation='Descriptive synthetic-split diagnostic; not native Gemma performance, an independent challenge or proof of shortcut elimination.')


def write_coverage_tables(rows,out,pilot,minimum_call):
    out.mkdir()
    def write(name,fields,items):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
    classes=Counter(r['labels']['D7'] for r in rows)
    calls=Counter(r['metadata']['call_category'] for r in rows)
    write('call_reason_distribution.csv',['call_category','minimum_target','actual','historical_2000_row_pilot_reference'],
          [dict(call_category=c,minimum_target=minimum_call,actual=calls[c],historical_2000_row_pilot_reference=pilot['categories'][c]) for c in sorted(calls)])
    write('D7_distribution.csv',['class','actual'],[{'class':c,'actual':n} for c,n in sorted(classes.items())])
    write('atomic_decisions.csv',['decision','true','false'],[dict(decision=d,true=sum(r['labels'][d] for r in rows),false=sum(not r['labels'][d] for r in rows)) for d in ('D1','D2','D3','D4','D5','D6')])
    boundaries=Counter(b for r in rows for b in r['metadata']['boundaries'])
    write('boundary_coverage.csv',['boundary','actual'],[dict(boundary=b,actual=boundaries[b]) for b in sorted(boundaries)])
    cross=[]
    for field in ('call_category','difficulty','recommendation_style','construction_recipe'):
        c=Counter((r['metadata'][field],r['labels']['D7']) for r in rows)
        cross.extend(dict(dimension=field,value=v,final_class=label,count=n) for (v,label),n in sorted(c.items()))
    for field,fn in (
        ('product_domain',lambda r:r['metadata']['scenario_product_domains'][0]),
        ('personalisation',lambda r:f"D3={r['labels']['D3']},D4={r['labels']['D4']}"),
        ('conversation_exchanges',lambda r:'single' if len(r['input']['customer_context'].splitlines())==1 else 'two_three' if len(r['input']['customer_context'].splitlines())<=5 else 'four_plus')):
        c=Counter((fn(r),r['labels']['D7']) for r in rows)
        cross.extend(dict(dimension=field,value=v,final_class=label,count=n) for (v,label),n in sorted(c.items()))
    write('cross_coverage.csv',['dimension','value','final_class','count'],cross)
    c=Counter((r['metadata']['call_category'],r['metadata']['difficulty']) for r in rows)
    write('call_reason_by_difficulty.csv',['call_category','difficulty','count'],[dict(call_category=a,difficulty=b,count=n) for (a,b),n in sorted(c.items())])
    lengths=[len(tokens(r['input']['agent_response'])) for r in rows]
    return dict(authored_case_families=len({r['metadata']['contrast_family'] for r in rows}),
        distinct_customer_contexts=len({r['input']['customer_context'] for r in rows}),
        distinct_response_texts=len({r['input']['agent_response'] for r in rows}),
        response_words=dict(min=min(lengths),max=max(lengths),mean=sum(lengths)/len(lengths)),
        difficulty_counts=dict(Counter(r['metadata']['difficulty'] for r in rows)),
        recommendation_style_counts=dict(Counter(r['metadata']['recommendation_style'] for r in rows)),
        disclaimer_counts_by_class={c:sum(r['metadata']['disclaimer'] for r in rows if r['labels']['D7']==c) for c in classes},
        recommendation_keywords_by_class={c:sum(bool(re.search(r'\b(?:recommend|should|best)\b',r['input']['agent_response'],re.I)) for r in rows if r['labels']['D7']==c) for c in classes},
        source_status='Fresh local authored construction cases; zero independent teacher or human approvals; no additional external generation API calls (this excludes the cost of this assistant session).')
