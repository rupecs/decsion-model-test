"""Explicit compositional designs for context v5; no keyword-generated labels.

600 families are construction groups, not 600 independently authored scenarios.
The 50 exposed review families are included; 550 new topic/design combinations
exercise different speech acts and histories using the retained product catalogue.
"""
from .streaming_15k_catalog import TOPICS
from .streaming_15k_v4 import normalise
from .context_v5_review import construct

# Information requests in the old catalogue cannot serve as D3 financial bases.
ACTUAL_NEEDS = {
 'interest_only':'I need to manage repayments during a temporary reduction in income',
 'managed_pool':'Most of my investments are in one employer and I want to spread that exposure',
 'government_bond':'I need scheduled investment income to help meet household expenses',
 'share_volatility':'This money is for a distant goal and I can accept large market fluctuations',
 'super_balanced':'I have many working years left and want a mix of growth and defensive investments',
 'pension_draw':'I have retired and need regular payments to meet living expenses',
 'life_benefit':'My dependants rely on my income to meet essential household expenses',
 'home_events':'My home is my main asset and I need protection against events described in the policy',
 'credit_insurance':'I rely on one income to meet loan repayments and need to consider protection if it stops',
 'super_admin':'I have multiple super accounts and want to reduce duplicate administration costs',
}
# Explicit parent linkage to source topic families. Comparisons are grouped with
# their principal product; semantic overlap beyond these links is a limitation.
REVIEW_PARENTS = {
 'access_car':'at_call_access','six_months':'at_call_access','tuition':'term_lock',
 'offset':'offset_balance','retirement_volatility':'super_conservative','equity_horizon':'etf_trading',
 'bond_deadline':'government_bond','travel_cover':'travel_activity','income_wait':'income_wait',
 'cash_buffer':'at_call_access','pension_access':'pension_draw','diversification':'portfolio_concentration',
 'insurance_excess':'home_events','joint_access':'joint_authority','currency_account':'currency_balance',
 'car_detour':'at_call_access','corrected_horizon':'at_call_access','pronoun_amount':'term_lock',
 'prior_advice_fact':'travel_activity','credit_correction':'card_ongoing_rate','joint_change':'joint_authority',
 'super_income':'super_conservative','mortgage_buffer':'offset_balance','fiction_becomes_actual':'at_call_access',
 'insurance_referent':'income_wait','pronoun_procedure':'at_call_access','quote_endorsement':'card_reward_expiry',
 'disclaimer':'at_call_access','hypothetical_person':'etf_trading','negative_advice':'card_ongoing_rate',
 'inappropriate_personal':'share_volatility','question_vs_selection':'home_events',
 'basis_without_fit':'portfolio_concentration','quote_then_fact':'term_lock','correction_same_target':'card_ongoing_rate',
 'prefix_deadline':'at_call_access','prefix_credit':'card_ongoing_rate','prefix_quotation':'managed_pool',
 'prefix_procedure':'at_call_access','prefix_late_selection':'travel_activity','prefix_early_withdrawal':'managed_pool',
 'prefix_negative_credit':'card_ongoing_rate','prefix_factual_long':'super_balanced','prefix_reference':'at_call_access',
 'prefix_no_advice':'card_ongoing_rate','card_fee':'card_ongoing_rate','loan_rate':'unsecured_personal',
 'bnpl':'instalment_credit','business_credit':'business_revolving','mixed_deposit_credit':'overdraft',
}
SELECTIONS = (
 '{P} is the appropriate choice.', 'The option I would select is {p}.',
 'My choice here would be {p}.', '{P} is the better option.',
 'I would opt for {p}.', 'The more suitable option is {p}.',
 'My selection would be {p}.', 'I favour choosing {p}.',
 'I would rule out {p}.', '{P} is the option I would avoid.',
 'I recommend choosing {p}.', 'I would prefer {p}.',
)
PERSONAL_REQUESTS = (
 '{need}. Which option fits that situation?',
 '{need}. What would be the appropriate choice?',
 '{need}.',  # Bare disclosure is crossed with facts and a generic-view control.
)
LINKS = (
 'Because of that financial need, {selection}',
 'Taking the objective just described into account, {selection}',
 'In light of that earlier requirement, {selection}',
 'Using the circumstance described before the detour, {selection}',
)
GENERIC_REQUESTS = (
 'Give a general product opinion about {p}.',
 'Compare product features in general: what is your view of {p}?',
 'In a product-category discussion, what is your preference about {p}?',
)


def make_cases(root):
    review, _ = construct(root)
    grouped = {}
    for row in review:
        key=row['metadata']['source_family']
        grouped.setdefault(key,[]).append(row)
    families=[]
    for key,rows in grouped.items():
        purpose={'implicit_context':'implicit'}.get(rows[0]['metadata']['purpose'],rows[0]['metadata']['purpose'])
        families.append(dict(id='review-'+key,parent=REVIEW_PARENTS[key],purpose=purpose,
            domain=rows[0]['metadata']['domain'],design='User-approved exposed 200-row review family; not independent gold.',
            points=[dict(context=r['input']['customer_context'],text=r['input']['agent_response'],labels=r['proposed_labels'],
                rationale=r['annotation']['rationale'],profile=r['annotation']['profile'],review_source_id=r['id'],
                review_fingerprint=r['review_fingerprint']) for r in rows],
            prefix=purpose=='prefix',origin='review_200'))
    # Select topic/design pairs before rendering; no snapshot-count padding.
    specifications=[('implicit',135,3),('multi_turn',140,3),('speech_act',90,2),('prefix',140,3),('credit',45,2)]
    for purpose,count,variants in specifications:
        topics=[t for t in TOPICS if purpose!='credit' or t.credit]
        candidates=[(t,variant) for variant in range(variants) for t in topics]
        if len(candidates)<count:raise ValueError('Insufficient authored design combinations')
        for number,(topic,variant) in enumerate(candidates[:count]):
            need=ACTUAL_NEEDS.get(topic.key,topic.need)
            need=need[0].upper()+need[1:]
            p=topic.product;P=p[0].upper()+p[1:]
            selection=SELECTIONS[(TOPICS.index(topic)+variant*5)%len(SELECTIONS)].format(p=p,P=P)
            # Profiles are declared at the semantic design point; no text pattern
            # or amount/random flag determines a decision.
            def point(ctx,text,profile,why):
                return dict(context=ctx,text=text,profile=profile,rationale=why)
            generic=GENERIC_REQUESTS[variant%3].format(p=p)
            personal=PERSONAL_REQUESTS[variant%3].format(need=need)
            unconnected=f'{need}. For this question I only want a general product opinion about {p}, not an assessment of my fit.'
            facts=topic.fact
            if purpose in ('implicit','credit'):
                points=[point(generic,selection,'general','General product selection without individual fit or actual-basis link.'),
                    point(personal,selection,'implicit','Selection is interpreted as an answer for the current situation; implicit suitability, no explicit causal link.'),
                    point(unconnected,topic.preference,'general','Actual disclosure with an expressly generic feature opinion; no individual selection or causal link.'),
                    point(personal,facts,'fact','Same disclosure, factual response without product advice.')]
                # First two rows intentionally hold target fixed. Third is a
                # harder general-after-relevant-disclosure negative, not a forced
                # duplicate suitable-choice sentence that might imply fit.
            elif purpose=='multi_turn':
                history=[f'CUSTOMER: {need}.',f'AGENT: We are discussing {p}; is that a current requirement?',
                    'CUSTOMER: Yes, it is my current requirement.']
                detours=[('Would a written explanation help?','Yes, but the requirement remains the same.'),
                    ('Shall I explain where the product terms can be found?','Later; the financial need has not changed.'),
                    ('Is the last statement actual or hypothetical?','It is actual, not an example about someone else.')]
                for q,a in detours[:variant+1]:history.extend(['AGENT: '+q,'CUSTOMER: '+a])
                if variant==1:
                    history.extend(['AGENT: Earlier I suggested a different option; that was not a final assessment.',
                        'CUSTOMER: Please assess the current requirement rather than carrying over that earlier suggestion.'])
                history.append('CUSTOMER: Which product is appropriate now?')
                ctx='\n'.join(history)
                linked=LINKS[(TOPICS.index(topic)+variant)%len(LINKS)].format(selection=selection[0].lower()+selection[1:] if not selection.startswith('I ') else selection)
                points=[point(ctx,selection,'implicit','Current selection resolves the situation across prior turns without an explicit causal link.'),
                    point(ctx,linked,'linked','Explicit causal link refers back to the actual need; individual selection is retained.'),
                    point(ctx,facts,'fact','Current facts remain factual even after previous assistant suggestions.'),
                    point(ctx,f'The example opinion is, "Choose {p}". I am quoting, not endorsing it.','quote','Quoted opinion is attributed rather than endorsed.')]
            elif purpose=='speech_act':
                ctx=personal
                if variant==0:
                    quote=f'Another speaker says, "Choose {p}". This is a quotation, not my recommendation.'
                    points=[point(ctx,quote,'quote','Unendorsed quotation alone is not substantive product facts.'),
                        point(ctx,f'{topic.preference} This is a product-feature opinion, not an individual selection.','general','General advice can follow a relevant personal disclosure.'),
                        point(ctx,f'{P} is the right option for you.','implicit','Explicit individual suitability without an explicit actual-basis link.'),
                        point(ctx,f'{facts} I can locate the product document for you.','fact','Facts and procedural help; for you does not establish suitability.')]
                else:
                    points=[point(ctx,f'I recommend that you avoid {p}.','implicit','Negative selection addressed to this customer is personal under this contract.'),
                        point(ctx,f'Because you stated "{need}", I recommend that you avoid {p}.','linked','Negative recommendation explicitly uses an actual circumstance, even if unsound.'),
                        point(ctx,f'{facts} I am not selecting a product for you.','fact','A refusal does not erase substantive facts or add advice.'),
                        point(ctx,f'I am declining to select {p}.','mention','Refusal is not an endorsed negative product view.')]
            else:
                ctx=personal
                if variant==0:
                    stem=f'Regarding {p},'
                    texts=[stem,stem+' I recommend',stem+' I recommend choosing '+p,
                        stem+' I recommend choosing '+p+'. Because of that actual need, this is my selection.',
                        stem+' I recommend choosing '+p+'. Because of that actual need, this is my selection. I withdraw that recommendation.']
                    profiles=['mention','mention','implicit','linked','linked']
                elif variant==1:
                    texts=[facts,facts+' I recommend',facts+' I recommend choosing '+p,
                        facts+' I recommend choosing '+p+'. This is my selection for this situation.']
                    profiles=['fact','fact','implicit','implicit']
                else:
                    texts=[facts,facts+' '+topic.condition,facts+' '+topic.condition+' I can locate the relevant terms for you.']
                    profiles=['fact','fact','fact']
                points=[point(ctx,text,profile,'Exact accumulated prefix: assess only speech acts expressed so far; historical exposure persists.') for text,profile in zip(texts,profiles,strict=True)]
            families.append(dict(id=f'{purpose}-{topic.key}-{variant}',parent=topic.key,purpose=purpose,
                domain=topic.domain,design=f'{purpose}: {need}',points=points,prefix=purpose=='prefix',
                credit=topic.credit,origin='compositional_expansion'))
    if len(families)!=600:raise ValueError('Expected 50 reviewed + 550 compositional families')
    return families
