"""Context-bound construction content for v10; no external or GPU generation."""
from pathlib import Path
import re
import runpy
from .streaming_15k_catalog import TOPICS
from .context_v5_review import digest

DISCLAIMERS=(
 'This reply is not a full assessment of a financial position.',
 'Please treat this as a discussion rather than a completed suitability assessment.',
 'This is not a substitute for a full review of the relevant circumstances.',
 'The scope of this reply is limited.',
 'Further assessment may be needed before a product decision.',
 'This response does not establish product eligibility.',
 'A complete personal assessment has not been carried out in this exchange.',
 'This discussion is not a completed assessment of all product risks.',
)
FITS=(
 'For you, {choice}',
 'You should {imperative}',
 'For you, I recommend {recommended}.',
 '{choice} That is my recommendation for you.',
 '{choice} I think that is right for you.',
 'My suggestion for you: {choice}',
 'If I were choosing for you, {choice}',
 'The best course for you is this: {choice}',
 '{choice} That would be my pick for you.',
 '{choice} That is what I would choose in your place.',
)
REFUSALS=(
 'I am not selecting {p} in this reply. Which feature should we discuss?',
 'I cannot recommend {p} solely from this exchange.',
 'I will leave the decision about {p} open rather than make a selection now.',
 'No choice of {p} is being made in this response.',
 'I am declining to endorse or reject {p} at this point.',
 'I have not reached a recommendation about {p}.',
 'I will not label {p} the best choice in this reply.',
 'Whether to select {p} is a decision I am leaving unresolved here.',
 'I am withholding a product view on choosing {p}.',
 'I am not advising for or against {p} in this answer.',
)


def asr_text(text):
    # Controlled punctuation/capitalisation loss only. Never drop negation,
    # alter a number, change a role boundary or simulate a different meaning.
    text=text.replace('’',"'")
    expansions={"can't":'cannot',"don't":'do not',"won't":'will not',"isn't":'is not',
        "aren't":'are not',"wasn't":'was not',"weren't":'were not',"doesn't":'does not',
        "didn't":'did not',"couldn't":'could not',"wouldn't":'would not',"shouldn't":'should not',
        "haven't":'have not',"hasn't":'has not',"hadn't":'had not',"i'm":'I am',"i've":'I have',"i'd":'I would'}
    for short,long in expansions.items():text=re.sub(r'\b'+re.escape(short)+r'\b',long,text,flags=re.I)
    text=re.sub(r'[^\w\s$%.,-]','',text)
    text=re.sub(r'(?<!\d)[.,](?!\d)|(?<=\d)[.,](?!\d)|(?<!\d),(?=\d)','',text)
    return text.lower()


def voice_text(text):
    for long,short in (('I am ',"I'm "),('I have ',"I've "),('I would ',"I'd "),('I cannot ',"I can't "),('do not ',"don't ")):
        text=text.replace(long,short)
    return text


def lower_start(text):
    return text if text.startswith(('I ',"I'm ","I've ","I'd ")) else text[0].lower()+text[1:]


def choice_forms(choice):
    if choice.startswith('I would '):imperative=choice[len('I would '):]
    elif choice.startswith('My choice would be '):imperative='choose '+choice[len('My choice would be '):]
    else:raise ValueError('An endorsed choice needs an explicit reviewed verb')
    action=imperative.rstrip('.')
    verbs={'choose':'choosing','favour':'favouring','shortlist':'shortlisting','avoid':'avoiding',
           'keep':'keeping','include':'including','select':'selecting','add':'adding'}
    first,_,rest=action.partition(' ')
    # Only the recommend-gerund cells use the returned gerund. The other cells
    # retain their exact authored polarity, including negative choices.
    original_first=first;original_rest=rest
    if first=='not':
        first,_,rest=rest.partition(' ')
        gerund='not '+verbs.get(first,first)+' '+rest
    else:gerund=verbs.get(first,first)+' '+rest
    recommended=original_rest if original_first in ('choose','favour','shortlist','include','select') else gerund
    return imperative,gerund,recommended


def contexts(case,topic):
    n=case['need']+'.';p=topic.product;f=topic.fact;q=case['question']
    explicit='I want a broad product comparison rather than a personal selection.'
    texts=[
      ['CUSTOMER: Could we discuss '+p+' and compare its features in general?'],
      ['CUSTOMER: Yeah, '+lower_start(n)+' What would you choose?'],
      ['CUSTOMER: Well, '+lower_start(n)],
      ['CUSTOMER: '+n,'AGENT: '+q,'CUSTOMER: '+explicit],
      ['CUSTOMER: Right, '+lower_start(n),'AGENT: '+q,'CUSTOMER: Please explain the features and then give a product view.'],
      ['CUSTOMER: An invented caller says: "'+n+'"','AGENT: Are we discussing a fictional example rather than your own circumstances?',
       'CUSTOMER: Yes, only the invented example. Explain '+p+' and any general product opinion.'],
      ['CUSTOMER: Okay, '+lower_start(n),'AGENT: '+case['action'],'CUSTOMER: Before we finish, could you explain '+p+'?',
       'AGENT: Which part needs explaining?','CUSTOMER: Start with the features, then give your current view.'],
      ['CUSTOMER: Look, '+lower_start(n),'AGENT: '+q,'CUSTOMER: I would like both parts explained.',
       'AGENT: '+f,'CUSTOMER: Thanks. I also want a comparison.',
       'AGENT: Are we returning to the requirement at the start?',
       'CUSTOMER: Yes, use that earlier requirement when giving the current response.'],
      ['CUSTOMER: Hi, '+lower_start(case['earlier'])+'.','AGENT: Thanks for explaining the original assumption.',
       'CUSTOMER: I need to correct that. '+n,'AGENT: '+q,
       'CUSTOMER: The correction is important to the comparison.','AGENT: '+f,
       'CUSTOMER: Right, please use the corrected circumstances for the response now.'],
      ['CUSTOMER: So, '+lower_start(n),'AGENT: Let us separate an explanation from a product selection.',
       'CUSTOMER: Yes, first explain '+p+'.','AGENT: '+f,
       'CUSTOMER: That helps with the terminology.','AGENT: Should the next answer return to the requirement already described?',
       'CUSTOMER: Yes, that earlier need is still the one I want addressed.'],
    ]
    names=('generic','actual','bare','explicit_generic','focused','fictional','prior_advice','history','correction','reference')
    for i,lines in enumerate(texts):
        voice=i in (1,2,4,6,7,8,9)
        asr=i==9 or (i==6 and int(digest(case['topic'])[:4],16)%3==0)
        if voice:lines=[line.split(': ',1)[0]+': '+voice_text(line.split(': ',1)[1]) if line.startswith('CUSTOMER: ') else line for line in lines]
        if asr:lines=[line.split(': ',1)[0]+': '+asr_text(line.split(': ',1)[1]) for line in lines]
        ctx='\n'.join(lines)
        actual=i not in (0,5)
        witness=voice_text(case['need']) if voice else case['need']
        if asr:witness=asr_text(witness)
        if actual:
            start=ctx.casefold().find(witness.casefold())
            if start<0:raise ValueError('Lost declared actual need')
            witness=ctx[start:start+len(witness)]
        else:witness=None
        yield dict(id=names[i],text=ctx,actual=actual,need=witness,
                   voice=voice,asr=asr,index=i)


def materialize(root):
    base_cases=runpy.run_path(str(root/'data/authored/streaming_context_v10/product_cases.py'))['CASES']
    variants=runpy.run_path(str(root/'data/authored/streaming_context_v10/case_variants.py'))['VARIANTS']
    opinions=runpy.run_path(str(root/'data/authored/streaming_context_v10/criterion_opinions.py'))['OPINIONS']
    choices=runpy.run_path(str(root/'data/authored/streaming_context_v10/short_selections.py'))['SELECTIONS']
    controls=runpy.run_path(str(root/'data/authored/streaming_context_v10/marker_controls.py'))['CONTROLS']
    criteria=runpy.run_path(str(root/'data/authored/streaming_context_v10/comparison_criteria.py'))['CRITERIA']
    implicit=runpy.run_path(str(root/'data/authored/streaming_context_v10/implicit_views.py'))['IMPLICIT']
    topics={t.key:t for t in TOPICS}
    if len(base_cases)!=60 or {c['topic'] for c in base_cases}!=set(topics) or any(set(x)!=set(topics) for x in (choices,variants,opinions,criteria,implicit)):
        raise ValueError('Incomplete explicit product-case specification')
    cases=[]
    for base in base_cases:
        topic=base['topic']
        cases.extend([
            {**base,'case_id':topic+'-a','variant':0,'criterion':criteria[topic],'criterion_opinion':opinions[topic][0]},
            {**base,**variants[topic],'case_id':topic+'-b','variant':1,'criterion_opinion':opinions[topic][1]},
        ])
    points=[]
    for ci,c in enumerate(cases):
        t=topics[c['topic']];choice=c.get('choice',choices[t.key]);p=t.product
        indirect=c.get('implicit',implicit[t.key]);preference=c.get('general',t.preference)
        imperative,gerund,recommended=choice_forms(choice)
        fit=FITS[ci%len(FITS)].format(choice=choice,imperative=imperative,gerund=gerund,recommended=recommended)
        # The same attribution appears before four different speech acts.
        conditional=(
            'If two offers are otherwise comparable, ',
            'With other conditions held equal, ',
            'When comparing otherwise similar offers, ',
            'For two offers with comparable terms, ',
            'If the remaining features are equivalent, ',
            'Assuming the other conditions are the same, ',
            'In an otherwise like-for-like comparison, ',
            'Where the remaining terms are comparable, ',
            'If the other product features match, ',
            'When other costs and conditions are unchanged, ',
        )[ci%10]+preference[0].lower()+preference[1:]
        for context in contexts({**c,'action':choice},t):
            selected=(0,1,2,5,8) if c['variant']==0 else (3,4,6,7,9)
            if context['index'] not in selected:continue
            idx=context['index'];actual=context['actual']
            # A short attributed utterance, not a bundle of class-marker phrases.
            quoted_view=(conditional,t.caution,fit,c['reason'],indirect,c['criterion_opinion'])[(idx+ci)%6]
            quotation=(('In an illustration, a speaker says: "{q}"',
                        'An illustrative speaker gives this opinion: "{q}"',
                        'A second speaker offers this view: "{q}"',
                        'A fictional adviser says in an example: "{q}"',
                        'The example contains the words "{q}" from another speaker.',
                        'An imagined conversation includes this opinion: "{q}"')[(ci+idx)%6]).format(q=quoted_view)
            implied=actual and context['id']!='explicit_generic'
            # Selection counterfactuals retain exactly the same target across
            # generic, fictional and actual contexts. No class is keyed by the
            # request wording: every context also receives all four classes.
            candidates=[
              ('facts',t.fact+' '+c['question'],'fact',False),
              ('own_quoted_fact','My answer is: "'+c['question']+' '+t.fact+'"','fact',False),
              ('own_quoted_choice','My answer is: "'+c['question']+' '+indirect+'"','implicit' if implied else 'general',False),
              ('own_quoted_question','My answer is: "'+c['question']+'"','mention',False),
              ('opinion',preference,'general',False),
              ('qualified_selection',choice+' '+t.condition,'implicit' if implied else 'general',True),
              ('criticism',t.caution,'general',False),
              ('conditional',conditional,'general',False),
              ('selection',choice,'implicit' if implied else 'general',False),
              ('fit',fit,'implicit',False),
              ('mixed_fit',t.condition+' '+fit,'implicit',True),
              ('clarification',c['question'],'mention',False),
              ('quote',quotation,'mention',False),
              ('quote_fact',quotation+' '+(t.fact if idx%2 else t.condition),'fact',False),
              ('quote_general',quotation+' '+t.fact+' '+preference,'general',True),
              ('quote_selection',quotation+' '+choice,'implicit' if implied else 'general',False),
              ('refusal',REFUSALS[idx].format(p=p),'mention',False),
              ('mixed_selection',t.fact+' '+choice,'implicit' if implied else 'general',True),
              ('selection_question',choice+' '+c['question'],'implicit' if implied else 'general',False),
            ]
            if actual:
                intro=(
                    'I am using '+c['criterion']+' as the comparison criterion.',
                    'The comparison here starts with '+c['criterion']+'.',
                    'For this discussion I have taken '+c['criterion']+' as the point to compare.',
                    'The point carried forward from the earlier requirement is '+c['criterion']+'.',
                    'I have framed this comparison around '+c['criterion']+'.',
                )[idx%5]
                candidates += [
                 ('linked',choice+' '+c['reason'],'linked',False),
                 ('basis_only',intro+' '+c['criterion_opinion'],'basis_only',False),
                 ('basis_fact',intro+' '+t.fact,'fact',False),
                 ('basis_question',intro+' '+c['question'],'mention',False)]
            for key,text,profile,mixed in candidates:
                if key=='mixed_fit' and context['index'] not in (3,4,6,7,8,9):continue
                disclaimer=int(digest(dict(topic=t.key,key=key))[:8],16)%10==0
                if disclaimer:text=DISCLAIMERS[ci%len(DISCLAIMERS)]+' '+text
                points.append(dict(family=c['case_id'],topic=t.key,context=context,text=text,profile=profile,
                    credit=t.credit,domain=t.domain,call=c['call'],point=context['id']+'.'+key,
                    mixed=mixed,disclaimer=disclaimer,case=c,
                    boundary_adjudication=profile=='basis_only'))
            for j,(profile,text) in enumerate(controls.get(t.key,()) if c['variant']==0 else ()):
                if profile=='linked' and not actual:continue
                points.append(dict(family=c['case_id'],topic=t.key,context=context,text=text,profile=profile,
                    credit=t.credit,domain=t.domain,call=c['call'],point=context['id']+'.marker'+str(j),
                    mixed=False,disclaimer=False,case=c,boundary_adjudication=False))
            # One real accumulating trajectory per topic, including a withdrawal.
            if context['id']=='history':
                chunks=[t.fact,choice,c['reason'],'I withdraw that product recommendation.']
                acc=[]
                for j,chunk in enumerate(chunks):
                    acc.append(chunk)
                    points.append(dict(family=c['case_id'],topic=t.key,context=context,text=' '.join(acc),
                        profile=('fact','implicit','linked','linked')[j],credit=t.credit,domain=t.domain,call=c['call'],
                        point='prefix.'+str(j),mixed=j>0,disclaimer=False,case=c,boundary_adjudication=False,
                        prefix=dict(checkpoint_index=j,total_checkpoints=4,is_final=j==3)))
            if context['id']=='correction':
                # Intentional refusal-to-advice drift: a declaration that no
                # choice is being made cannot erase later endorsed advice.
                # References and product/need relationships remain grounded.
                chunks=[REFUSALS[(ci//2)%len(REFUSALS)].format(p=p),t.fact,
                        preference,choice,c['reason'],
                        'I withdraw that product recommendation.']
                acc=[]
                for j,chunk in enumerate(chunks):
                    acc.append(chunk)
                    points.append(dict(family=c['case_id'],topic=t.key,context=context,text=' '.join(acc),
                        profile=('mention','fact','general','implicit','linked','linked')[j],credit=t.credit,domain=t.domain,call=c['call'],
                        point='drift.'+str(j),mixed=j>=2,disclaimer=False,case=c,boundary_adjudication=False,
                        prefix=dict(scenario_id='drift',checkpoint_index=j,total_checkpoints=6,is_final=j==5)))
            if context['id']=='generic':
                for j,text in enumerate((preference,preference+' I withdraw that product recommendation.')):
                    points.append(dict(family=c['case_id'],topic=t.key,context=context,text=text,
                        profile='general',credit=t.credit,domain=t.domain,call=c['call'],point='generic_prefix.'+str(j),
                        mixed=False,disclaimer=False,case=c,boundary_adjudication=False,
                        prefix=dict(scenario_id='generic_prefix',checkpoint_index=j,total_checkpoints=2,is_final=j==1)))
    return points


def service_points(root):
    services=runpy.run_path(str(root/'data/authored/streaming_context_v10/service_cases.py'))['SERVICES']
    result=[]
    for si,s in enumerate(services):
        histories=[
          ['CUSTOMER: '+s['customer']],
          ['CUSTOMER: Right, '+lower_start(s['customer']),
           'AGENT: Would you like the existing process explained?',
           'CUSTOMER: Yes, please explain the process and what I can do next.'],
          ['CUSTOMER: '+s['customer'],'AGENT: '+s['fact'],
           'CUSTOMER: I understand that part. I still need help with the next step.',
           'AGENT: We can discuss the process without sharing private credentials.',
           'CUSTOMER: Yes, I want to keep the private details out of this conversation.',
           'AGENT: Should the next reply focus on the existing issue?',
           'CUSTOMER: Yes, please return to the issue I described.'],
          ['CUSTOMER: '+s['customer'],'AGENT: We can start by explaining the process.',
           'CUSTOMER: Yep, that is what I need.','AGENT: Should we keep the discussion to the existing request?',
           'CUSTOMER: Yes, explain the existing request rather than choosing a new product.'],
        ]
        for hi,lines in enumerate(histories):
            if hi in (1,3):lines=[line.split(': ',1)[0]+': '+voice_text(line.split(': ',1)[1]) if line.startswith('CUSTOMER: ') else line for line in lines]
            if hi==2:lines=[line.split(': ',1)[0]+': '+asr_text(line.split(': ',1)[1]) for line in lines]
            ctx=dict(id='service'+str(hi),text='\n'.join(lines),actual=False,need=None,
                     voice=hi in (1,3),asr=hi==2,index=hi)
            for j,(text,profile) in enumerate([
                (s['fact'],'fact'),(s['procedure'],'fact'),(s['fact']+' '+s['procedure'],'fact'),
                (s['question'],'mention'),(s['alternative'],'mention'),
                (s['fact']+' '+s['question'],'fact')]):
                disclaimer=int(digest(dict(service=s['id'],point=j))[:8],16)%10==0
                if disclaimer:text=DISCLAIMERS[si%len(DISCLAIMERS)]+' '+text
                result.append(dict(family='service-'+s['id'],topic='service-'+s['id'],
                    context=ctx,text=text,profile=profile,credit=False,domain='service',call=s['call'],
                    point=ctx['id']+'.'+str(j),mixed=False,disclaimer=disclaimer,
                    case=dict(need='',action='',reason='Operational service request, without a product selection.'),
                    boundary_adjudication=False,product_scope=s['product_scope']))
    return result


def arithmetic_points():
    """Three materially different personalised fact tasks, not numeric clones."""
    specs=[
      ('simple_interest','at_call_access','C08',
       'For an illustration of a savings account, use my $8,000 balance and a hypothetical 3% simple annual rate for one year. Ignore compounding, tax and fees. What is the interest?',
       'Under those hypothetical simple-interest assumptions, $8,000 multiplied by 0.03 is $240 for one year, before tax and fees.'),
      ('fee_comparison','transaction_fees','C01',
       'I make ten of these account transactions each month. For a hypothetical comparison, account A has a $6 monthly fee plus $0.20 per transaction; account B has no monthly fee and charges $0.50 per transaction. What are the totals?',
       'On those hypothetical schedules and ten transactions, account A totals $8 for the month: $6 plus $2. Account B totals $5. That arithmetic does not select an account.'),
      ('reserved_balance','at_call_access','C01',
       'For this illustration, my savings account has $7,200 and I have already decided to reserve $1,800 for the quarterly bills. Ignoring all other transactions, what remains?',
       'Subtracting the already-reserved $1,800 from $7,200 leaves $5,400 in this illustration. That calculation ignores other transactions as requested.'),
    ]
    for key,topic,call,customer,text in specs:
        yield dict(family='calculation-'+key,topic=topic,
            context=dict(id='calculation',text='CUSTOMER: '+customer,actual=True,need=customer,voice=False,asr=False,index=0),
            text=text,profile='fact',credit=False,domain='deposit',call=call,point='calculation.'+key,
            mixed=False,disclaimer=False,case=dict(need=customer,action='',reason='Arithmetic using stated hypothetical assumptions, without a product selection.'),
            boundary_adjudication=False,personalized_fact=True)
