"""Authored speech-act banks and explicit customer-basis types for v4.

Combinations are synthetic, not independently written or verified examples.
Frame IDs are split-owned; training contains eight frames per recipe.
"""
from .streaming_15k_catalog import TOPICS
import re
from .streaming_15k_v2 import NEEDS as OLD_NEEDS, context as old_context, reported_need, stable
from .streaming_15k_long import LONG, SECURITY as LONG_SECURITY

FRAME_POOLS = {'train': tuple(range(8)), 'validation': (8, 9),
               'calibration': (10, 11), 'test': (12, 13)}
ACTS = ('mechanism', 'condition', 'document_help', 'clarification', 'refusal',
        'quotation', 'quote_endorsement', 'personal_positive', 'personal_negative',
        'unsupported_suitability', 'general_positive', 'general_negative', 'mixed',
        'withdrawal', 'security_procedure', 'nonproduct_question',
        'incremental_recommendation', 'incremental_document', 'context_to_fact',
        'late_general', 'late_personal', 'late_negative', 'late_endorsement', 'late_mid_sentence',
        'long_facts', 'long_conditions', 'long_document', 'long_quotation', 'long_quotation_alt',
        'early_general', 'basis_positive', 'basis_negative', 'basis_explanation', 'basis_document')
NEEDS = {**OLD_NEEDS, 'insurance': OLD_NEEDS['insurance'][:-1]+(
    'I have booked overseas activities and need insurance cover for them',)}
EXTRA_BASIS = {
    'credit': (
        'my contract income arrives irregularly and I need repayments I can manage between jobs',
        'I need to repay borrowing before a planned reduction in working hours',
        'I am carrying debt across several statements and want to reduce ongoing interest costs',
        'I need access to borrowing for staged equipment purchases rather than a single payment',
        'my household has one income and I need predictable repayment commitments',
        'I plan to make additional repayments from seasonal business receipts',
    ),
    'deposit': (
        'I need an emergency reserve I can withdraw without advance notice',
        'I am saving for school fees due at a known date',
        'my income is seasonal and I need cash available between payments',
        'I want to separate everyday spending money from savings for a planned expense',
        'I need a cash reserve for repairs while keeping other savings aside',
        'I am setting aside money for a home deposit and cannot accept investment losses',
    ),
    'payment': (
        'I pay recurring household bills and need to track which payments have been authorised',
        'my business makes occasional overseas payments and needs to control transaction costs',
        'I need to make everyday purchases using money already in my account',
        'I share household bills and need clear authority over payments',
        'I receive irregular payments and need to organise recurring bill dates',
        'I need access to payment records for my business bookkeeping',
    ),
    'investment': (
        'I need access to part of my investments for a planned expense',
        'most of my investments are in my employer and I want to reduce concentration',
        'I depend on investment income to meet household costs',
        'I am saving over a long period and can accept changes in market value',
        'I have a near-term spending commitment and cannot risk losing its funding',
        'I want to spread my investment exposure across different assets',
    ),
    'super': (
        'I am approaching retirement and need to organise an income stream',
        'I have several super accounts and want to reduce duplicate administration costs',
        'I have many working years ahead and can accept investment fluctuations',
        'I rely on retirement income for recurring household expenses',
        'I am concerned about changes in my retirement savings close to retiring',
        'I need to organise retirement savings after a change in working hours',
    ),
    'insurance': (
        'I support dependants who rely on my income for household bills',
        'I am self-employed and do not have paid sick leave to cover time away from work',
        'I have booked overseas activities and need cover for those activities',
        'I own expensive equipment that I use to earn my income',
        'I share a vehicle with another household member who regularly drives it',
        'I need to protect belongings that I could not readily replace from savings',
    ),
}

BRIDGES = (
    'The document identifies the conditions applicable to that feature; it does not establish approval or individual suitability.',
    'The feature wording and the conditions describe different parts of this arrangement.',
    'A product description does not settle the terms of an individual application.',
    'The current terms are needed to identify which conditions apply to an actual arrangement.',
    'The name of the product is not a complete statement of its conditions.',
    'Reading a feature summary does not establish an entitlement under the actual terms.',
    'This describes the feature; a specific arrangement has its own applicable wording.',
    'A feature heading is an index to the wording rather than a complete description of the terms.',
    'The product wording supplies conditions that a short description can omit.',
    'The actual agreement identifies the obligations associated with that feature.',
    'A feature description and a record of an individual arrangement serve different purposes.',
    'The terms explain the arrangement without determining an individual application outcome.',
    'The explanation of a feature does not by itself identify every condition in the agreement.',
    'The applicable document is required to establish the conditions of a particular arrangement.',
)

def sentence_units(text):
    return [m.group() for m in re.finditer(r'.+?(?:[.!?][\"\u201d\u2019]?(?=\s|$)|$)', text, re.S)]

def repeated_sentences(text):
    units = [' '.join(s.strip().split()).casefold() for s in sentence_units(text)
             if re.search(r'[.!?][\"\u201d\u2019]?$', s.strip())]
    return sorted({s for s in units if units.count(s)>1})

def clean_repetition(points):
    """Remove repeated complete sentences; never rewrite an earlier prefix.

    A redundant checkpoint is dropped, not padded to retain an arbitrary length.
    An unsafe prefix edit rejects the candidate rather than changing its gold.
    """
    accepted = []
    for point in points:
        seen, units = set(), []
        for unit in sentence_units(point['text']):
            key = ' '.join(unit.strip().split()).casefold()
            complete = bool(re.search(r'[.!?][\"\u201d\u2019]?$', unit.strip()))
            if complete and key in seen:
                continue
            if complete:
                seen.add(key)
            units.append(unit)
        cleaned = ''.join(units).strip()
        if accepted and not cleaned.startswith(accepted[-1]['text']):
            return []
        if accepted and cleaned == accepted[-1]['text']:
            continue
        point = dict(point, text=cleaned)
        if point['evidence'] not in cleaned:
            return []
        accepted.append(point)
    return accepted if len(accepted)>=3 else []

# These are authored semantic types, not keyword-derived gold decisions.
INFORMATION_TOPICS = {
    'interest_only', 'managed_pool', 'government_bond', 'share_volatility',
    'super_balanced', 'pension_draw', 'life_benefit', 'home_events', 'credit_insurance',
}
FINANCIAL_TOPICS = {
    'card_ongoing_rate', 'cash_advance', 'consolidation', 'overdraft', 'redraw_feature',
    'rental_property', 'business_revolving', 'at_call_access', 'bonus_conditions',
    'offset_balance', 'transaction_fees', 'remittance_cost', 'merchant_cost',
    'super_admin', 'income_wait', 'car_excess', 'contents_limits',
}
OBJECTIVE_TOPICS = {
    'card_reward_expiry', 'balance_transfer', 'purchase_promotion', 'credit_limit',
    'unsecured_personal', 'secured_car', 'instalment_credit', 'fixed_mortgage',
    'variable_mortgage', 'equity_line', 'construction_draws', 'equipment_security',
    'margin_buffer', 'guaranteed_loan', 'reverse_mortgage', 'bridging_period',
    'notice_access', 'term_lock', 'term_ladder', 'debit_link', 'joint_authority',
    'junior_access', 'business_access', 'currency_balance', 'wallet_credential',
    'scheduled_payment', 'etf_trading', 'index_tracking', 'corporate_credit',
    'portfolio_concentration', 'property_fund', 'super_conservative',
    'travel_activity', 'settlement_cash',
}
TOPIC_BASIS = {**dict.fromkeys(INFORMATION_TOPICS, 'information_request'),
               **dict.fromkeys(FINANCIAL_TOPICS, 'financial_situation'),
               **dict.fromkeys(OBJECTIVE_TOPICS, 'objective_or_need')}
assert set(TOPIC_BASIS) == {t.key for t in TOPICS}
assert len(INFORMATION_TOPICS)+len(FINANCIAL_TOPICS)+len(OBJECTIVE_TOPICS) == len(TOPICS)
DOMAIN_TYPES = {
    'credit': ('objective_or_need', 'objective_or_need', 'financial_situation', 'objective_or_need', 'information_request'),
    'deposit': ('objective_or_need', 'information_request', 'objective_or_need', 'objective_or_need', 'financial_situation'),
    'payment': ('financial_situation', 'objective_or_need', 'objective_or_need', 'financial_situation', 'information_request'),
    'investment': ('objective_or_need', 'information_request', 'objective_or_need', 'objective_or_need', 'information_request'),
    'super': ('objective_or_need', 'information_request', 'information_request', 'information_request', 'objective_or_need'),
    'insurance': ('information_request', 'objective_or_need', 'information_request', 'information_request', 'objective_or_need'),
}

def basis(topic, case):
    if case >= 12:
        return EXTRA_BASIS[topic.domain][case-12], 'objective_or_need'
    if case >= 6:
        return None, 'none'
    if case == 0:
        return topic.need, TOPIC_BASIS[topic.key]
    return NEEDS[topic.domain][case-1], DOMAIN_TYPES[topic.domain][case-1]

def context(topic, case):
    if case >= 12:
        need, _ = basis(topic, case)
        text = f'CUSTOMER: {need[0].upper()+need[1:]}. Can we discuss {topic.product} and its features?'
    else:
        text, _ = old_context(topic, case)
    if topic.domain == 'insurance' and case == 5:
        text = text.replace(OLD_NEEDS['insurance'][-1], NEEDS['insurance'][-1])
    # Every speech act sees these same two-part requests, including advice/facts.
    if case % 3 == 1:
        text += ' I also need help signing in securely to view the document.'
    return text, basis(topic, case)[0]

POSITIVE = (
    'I recommend {p} for you.', 'You should choose {p}.',
    '{P} is my preferred option for you.', 'Given your situation, I would select {p} for you.',
    'My recommendation to you is to take {p}.', 'I think {p} suits you.',
    'The choice I favour for you is {p}.', 'In your place, I would go with {p}.',
    'I would steer you towards {p}.', 'For you, my selection is {p}.',
    '{P} would be the option I pick for you.', 'My view is that you should take {p}.',
    'If I were choosing for you, I would choose {p}.', 'I suggest you opt for {p}.',
)
NEGATIVE = (
    'I recommend that you avoid {p}.', 'You should not choose {p}.',
    '{P} is not suitable for you.', 'For your situation, I would rule out {p}.',
    'My advice to you is to pass on {p}.', 'I think {p} is a poor fit for you.',
    'The option I would exclude for you is {p}.', 'In your place, I would not take {p}.',
    'I would steer you away from {p}.', 'For you, I advise against {p}.',
    '{P} would not be my choice for you.', 'My view is that you should avoid {p}.',
    'If I were choosing for you, I would reject {p}.', 'I suggest you do not take {p}.',
)
REFUSAL = (
    'I will not choose {p} on your behalf.', 'I am declining to select {p} for you.',
    'I cannot make your decision about {p}.', 'I am leaving the choice of {p} to you.',
    'Whether to take {p} is not a selection I am making here.', 'I am not offering a view on whether you should take {p}.',
    'I have no recommendation to give you about choosing {p}.', 'I am not deciding for or against {p}.',
    'I am withholding a product selection about {p}.', 'I am not endorsing or rejecting {p} for you.',
    'I will leave the decision on {p} open.', 'I am not expressing a preference about taking {p}.',
    'I am declining to advise on your selection of {p}.', 'I will not settle the choice of {p} for you.',
)
QUOTATION = (
    'An example of someone else\'s opinion about {p} is: "{op}"',
    'The following opinion about {p} is quoted, not endorsed: "{op}"',
    'A fictional speaker says about {p}, "{op}"',
    'Someone in the example comments on {p}: "{op}"',
    'These are the words of another speaker about {p}: "{op}"',
    'For illustration, here is a quoted view about {p}: "{op}"',
    'A sample opinion concerning {p} reads, "{op}"',
    'I am quoting a hypothetical view about {p}: "{op}"',
    'In the example about {p}, the speaker says, "{op}"',
    'Consider another person\'s words on {p}: "{op}"',
    'The illustrative comment about {p} is, "{op}"',
    'This is a quotation of an opinion on {p}: "{op}"',
    'A made-up speaker offers this view on {p}: "{op}"',
    'Here is an attributed opinion about {p}, rather than my selection: "{op}"',
)
FACT = (
    '{fact}', 'One feature of {p} is this: {fact}',
    'For {p}, the document describes this mechanism. {fact}',
    'The operation of {p} includes the following. {fact}',
    'Here is a feature explanation for {p}. {fact}',
    'When reading about {p}, this is the mechanism to note. {fact}',
    'To explain {p}: {fact}', 'About the way {p} works: {fact}',
    'This is how {p} operates under its wording. {fact}',
    'The feature being described for {p} works as follows. {fact}',
    'For your question on {p}, the factual point is: {fact}',
    'The explanation of {p} begins with its mechanism. {fact}',
    'In describing {p}, I mean this. {fact}',
    'A point about how {p} works: {fact}',
)
DOCUMENT = (
    'Open the terms document for {p} and use its feature headings.',
    'For {p}, find the conditions using the document contents list.',
    'To locate the wording for {p}, select the relevant document heading.',
    'The feature section of the terms for {p} contains the applicable wording.',
    'To locate the conditions of {p} for you, the service team can use the document index.',
    'I recommend opening the document for {p} at the feature section.',
    'Read the feature heading in the terms for {p} to find the clause.',
    'For you to read the conditions of {p}, start with the document contents list.',
    'Use the terms index to locate the feature described for {p}.',
    'Find {p} in the document list, then open the feature heading.',
    'I can show the feature section for you: open the terms for {p} and follow the relevant heading.',
    'I suggest using the document headings to find the conditions for {p}.',
    'For {p}, the contents page directs you to the applicable feature clause.',
    'Locate the terms for {p}; its index points to the relevant conditions.',
)
CLARIFICATION = (
    'About {p}, which feature do you mean?', 'For {p}, is your question about its operation or its conditions?',
    'Which aspect of {p} are you asking about?', 'On {p}, do you mean the mechanism or the written restrictions?',
    'What would you like clarified about {p}?', 'Is there a particular clause about {p} you want to discuss?',
    'When you mention {p}, what part is your question about?', 'Could you identify the feature of {p} you mean?',
    'Concerning {p}, which point should we address?', 'What is the unresolved part of your question about {p}?',
    'Would you like to discuss the operation or the terms of {p}?', 'Regarding {p}, where should the explanation begin?',
    'For the question about {p}, what wording are you referring to?', 'Which point on {p} would you like explained first?',
)
CONDITION = (
    'The conditions for {p} include the following: ',
    'For {p}, this condition applies: ',
    'A restriction described for {p} is this: ',
    'When reading the terms for {p}, note that ',
    'On the conditions of {p}, the relevant point is: ',
    'For the feature terms of {p}, the document states: ',
    'One condition attached to {p} is as follows: ',
    'The wording about {p} specifies that ',
    'In the conditions for {p}, this is the qualification: ',
    'For {p}, the stated limitation is this: ',
    'A point in the terms governing {p} is: ',
    'For the restrictions on {p}, the explanation is: ',
    'Under the terms describing {p}, ',
    'This is a condition concerning {p}: ',
)
START = (
    'I recommend', 'You should consider', 'My recommendation is', 'I would choose',
    'I suggest choosing', 'The option I favour is', 'I would recommend', 'My suggested selection is',
    'I favour', 'The product I suggest is', 'I advise choosing', 'My preference is',
    'I would opt for', 'What I recommend is',
)
DOCUMENT_START = (
    'I recommend that you', 'You should', 'To find the clause,',
    'My suggestion is to', 'I suggest that you', 'For the document search,',
    'I would suggest you', 'The next document step is to',
    'To locate the wording,', 'For the terms lookup,', 'I advise you to',
    'The way to find the clause is to', 'To read the conditions,', 'I recommend you',
)
DOCUMENT_ACTION = (
    'open the terms for {p} and find the relevant feature heading.',
    'read the feature section in the document describing {p}.',
    'locate the clause for {p} through the document index.',
    'use the contents list in the terms for {p} to locate the condition.',
    'find the feature description in the documentation for {p}.',
    'select the feature heading in the terms for {p}.',
    'turn to the conditions section of the document for {p}.',
    'follow the contents entry for the relevant feature of {p}.',
    'inspect the feature headings in the documentation for {p}.',
    'access the document for {p} through its entry in the document list.',
    'search the contents of the terms for the feature of {p}.',
    'look at the document index for the conditions of {p}.',
    'refer to the feature section in the terms describing {p}.',
    'go to the heading for the feature in the document for {p}.',
)
GENERAL = (
    '{op}', 'As a comparison of product features, {op_lower}',
    'My view of the product features is this: {op}',
    'On the merits of these features: {op}',
    'When comparing the options, I would put it this way. {op}',
    'The product comparison I favour is: {op}',
    'Here is my opinion of the choice between these features. {op}',
    'My preference concerns the features themselves: {op}',
    'Looking at product design, {op_lower}',
    'As a product judgement, {op_lower}',
    'I would use this comparison of product features: {op}',
    'The features lead me to this opinion. {op}',
    'My view on the alternatives is as follows. {op}',
    'In a feature comparison, {op_lower}',
)
TAILS = (
    'Which clause would you like explained?', 'Would you like to specify what the team can explain?',
    'Would you prefer a brief explanation or a detailed one?', 'Do you have a question about the document?',
    'Could you identify a feature to discuss?', 'What part would you like the service team to explain?',
    'Would you like to look at the conditions?', 'Which detail should we address next?',
    'Is there a term you want explained?', 'Where would you like the explanation to begin?',
    'What question do you have about the terms?', 'Would a clause explanation help?',
    'Which document section should we discuss?', 'Do you want to ask about a feature?',
)
SECURITY = (
    'Keep your password private.', 'Do not disclose a one-time sign-in code to a caller.',
    'I recommend using the official sign-in page.', 'Use the official app for the reset process.',
    'You should keep authentication details to yourself.', 'Open the bank service through its official website.',
    'I suggest verifying the contact channel independently.', 'Decline unexpected remote-access requests.',
    'Keep the sign-in code within the expected authentication flow.', 'Find the reset link in the official sign-in screen.',
    'Use a contact number from a bank source you already trust.', 'Do not share your login password.',
    'Contact support using an independently verified route.', 'Do not give a caller remote access to sign in.',
)

def frame_id(topic, case, act, split):
    pool = FRAME_POOLS[split]
    return pool[int(stable(topic.key+':'+str(case)+':'+act+':frame')[:8], 16) % len(pool)]

def render(topic, case, act, split):
    """Authored checkpoint semantics. Future continuation never assigns a label."""
    v = frame_id(topic, case, act, split)
    need, kind = basis(topic, case)
    use_basis = kind in ('objective_or_need', 'financial_situation')
    personal = ('credit_personal' if topic.credit else 'personal') if use_basis else ('credit_suitability' if topic.credit else 'suitability')
    suitability = 'credit_suitability' if topic.credit else 'suitability'
    general = 'credit_general' if topic.credit else 'general'
    values = dict(p=topic.product, P=topic.product[0].upper()+topic.product[1:],
                  fact=topic.fact, op=topic.preference, op_lower=topic.preference[0].lower()+topic.preference[1:])
    fmt = lambda bank: bank[v].format(**values)
    text = ''
    points = []
    def add(chunk, profile, evidence=None, **flags):
        nonlocal text
        if text.endswith(',') and chunk[0].isupper() and not chunk.startswith(('I ', 'I\'m ', 'I\'ll ')):
            original = chunk
            chunk = chunk[0].lower()+chunk[1:]
            if evidence == original:
                evidence = chunk
        text += (' ' if text and chunk[0] not in ',.:;?!' else '')+chunk
        points.append(dict(text=text, profile=profile, evidence=evidence or chunk, flags=flags))
    def tailored_tail(negative=False):
        if use_basis:
            return (
                f'I base that recommendation on your statement, "{need}".',
                f'You told me, "{need}"; that is my reason for this advice.',
                f'My reason for advising that is what you said: "{need}".',
                f'That advice takes account of your statement, "{need}".',
                f'I am using the circumstance you described: "{need}".',
                f'Your statement, "{need}", is the basis of my selection advice.',
                f'In making that recommendation I have relied on this: "{need}".',
                f'I am linking the recommendation to your own words, "{need}".',
                f'The reason behind my advice is your description: "{need}".',
                f'I have used what you told me, "{need}", to make that recommendation.',
                f'My selection advice follows from your statement, "{need}".',
                f'That is my advice in light of what you said: "{need}".',
                f'I am taking your statement, "{need}", into account in advising you.',
                f'I have made that recommendation using this detail from you: "{need}".',
            )[v]
        # No fabricated customer basis, and no affirmative ending on a negative
        # recommendation. Topic facts add useful content instead of restatement.
        return topic.condition if negative else topic.fact
    if act in ('mechanism', 'condition', 'context_to_fact'):
        if act == 'mechanism':
            add(fmt(FACT), 'factual', topic.fact)
        elif act == 'condition':
            add(fmt(CONDITION).rstrip(), 'product_fragment', topic.product)
        else:
            add(('Given your', 'Regarding your', 'In response to your', 'To address your')[v % 4], 'fragment')
        if act == 'condition':
            # Keep the grammatical seam after a colon or a conjunction.
            clause = topic.condition
            if text.endswith('that'):
                clause = clause[0].lower()+clause[1:]
            add(clause, 'factual', clause)
        elif act == 'context_to_fact':
            fact = topic.fact[0].lower()+topic.fact[1:]
            add(f'question about {topic.product}, '+fact, 'factual', fact)
        else:
            add(topic.condition, 'factual')
        add((topic.fact+' ' if act == 'condition' else '')+fmt(DOCUMENT), 'factual')
    elif act in ('document_help', 'incremental_document'):
        add((f'Documentation for {topic.product}' if act == 'document_help' else DOCUMENT_START[v]),
            'product_fragment' if act == 'document_help' else 'fragment')
        add((': '+fmt(DOCUMENT) if act == 'document_help' else
             fmt(DOCUMENT_ACTION)), 'factual')
        add(topic.condition, 'factual')
    elif act == 'clarification':
        question = fmt(CLARIFICATION)
        add(f'Your question about {topic.product}', 'product_fragment', topic.product)
        add(': '+question, 'product_other', question)
        add(TAILS[(v+3) % len(TAILS)], 'product_other')
    elif act == 'refusal':
        add(fmt(REFUSAL), 'product_other')
        add(fmt(TAILS), 'product_other')
        add(TAILS[(v+5) % len(TAILS)], 'product_other')
    elif act in ('quotation', 'quote_endorsement'):
        add(fmt(QUOTATION), 'product_other')
        if act == 'quotation':
            add('I am illustrating the wording, not endorsing that opinion.', 'product_other')
            # Facts after a quotation can change Other to factual, without advice.
            if case % 2:
                add(topic.fact, 'factual')
            else:
                add('The quoted view remains attributed to the illustrative speaker.', 'product_other')
        else:
            add(fmt(POSITIVE), suitability)
            add(tailored_tail(), personal)
    elif act in ('personal_positive', 'personal_negative'):
        negative = act == 'personal_negative'
        add(f'On the question of {topic.product},', 'product_fragment')
        add(fmt(NEGATIVE if negative else POSITIVE), suitability)
        add(tailored_tail(negative), personal)
    elif act == 'unsupported_suitability':
        add(fmt(POSITIVE), suitability)
        add(topic.fact, suitability, points[0]['evidence'])
        add(topic.condition, suitability, points[0]['evidence'])
    elif act in ('general_positive', 'general_negative'):
        if act == 'general_negative':
            values.update(op=topic.caution, op_lower=topic.caution[0].lower()+topic.caution[1:])
        add(fmt(GENERAL), general)
        add(topic.fact, general, points[0]['evidence'])
        add(topic.condition, general, points[0]['evidence'])
    elif act == 'mixed':
        add(fmt(FACT), 'factual')
        add(fmt(GENERAL), general)
        add(fmt(POSITIVE), suitability)
    elif act == 'withdrawal':
        stance = general if case % 2 else suitability
        add(fmt(GENERAL) if case % 2 else fmt(POSITIVE), stance)
        add(topic.condition, stance, points[0]['evidence'])
        add('I withdraw that recommendation; I have not assessed whether it is suitable for you.',
            stance, points[0]['evidence'], recommendation_withdrawn=True)
    elif act == 'security_procedure':
        if case % 3 != 1:
            return []  # Only the shared two-part security/product customer contexts.
        add('For the sign-in part of your question: '+SECURITY[v], 'procedure')
        add(SECURITY[(v+3) % len(SECURITY)], 'procedure')
        add(SECURITY[(v+5) % len(SECURITY)], 'procedure')
    elif act == 'nonproduct_question':
        add(('Before I continue,', 'To choose the level of detail,', 'For the explanation,', 'Before the explanation,')[v % 4], 'fragment')
        add(fmt(TAILS), 'other')
        add(TAILS[(v+7) % len(TAILS)], 'other')
    elif act == 'incremental_recommendation':
        add(START[v], 'fragment')
        add(topic.product, general)
        add('for you. '+tailored_tail(), personal)
    elif act.startswith('basis_'):
        if not use_basis:
            return []
        add(fmt(NEGATIVE if act == 'basis_negative' else POSITIVE)+' '+tailored_tail(act == 'basis_negative'), personal)
        add(topic.fact, personal, points[0]['evidence'])
        add(topic.condition, personal, points[0]['evidence'])
        add(BRIDGES[v], personal, points[0]['evidence'])
        add(fmt(DOCUMENT), personal, points[0]['evidence'])
        if act in ('basis_explanation', 'basis_document'):
            add(LONG[topic.domain][v % 4], personal, points[0]['evidence'])
    elif act in ('long_quotation', 'long_quotation_alt'):
        add(fmt(QUOTATION), 'product_other')
        add('That is the illustrative speaker\'s view, not my recommendation.', 'product_other')
        add(f'A second speaker offers another opinion about {topic.product}:', 'product_other')
        add('"'+topic.caution+'"', 'product_other')
        if act == 'long_quotation_alt':
            add('The second opinion is also attributed to its speaker.', 'product_other')
        add('I am quoting these views without endorsing either product selection.', 'product_other')
    elif act in ('long_facts', 'long_conditions', 'long_document', 'early_general'):
        stance = general if act == 'early_general' else 'factual'
        add(fmt(GENERAL) if act == 'early_general' else fmt(FACT), stance)
        add(topic.condition, stance, points[0]['evidence'])
        add(fmt(DOCUMENT), stance, points[0]['evidence'])
        add(BRIDGES[v], stance, points[0]['evidence'])
        if act in ('long_conditions', 'long_document') or (act == 'early_general' and case % 2):
            add(LONG[topic.domain][v % 4], stance, points[0]['evidence'])
        add('The terms and the document date identify the version of the wording being discussed.', stance, points[0]['evidence'])
    elif act.startswith('late_'):
        add(fmt(FACT), 'factual')
        add(topic.condition, 'factual')
        add(fmt(DOCUMENT), 'factual')
        # Each prefix is assessed as it stands, including a dangling advice stem.
        if act == 'late_endorsement':
            add(fmt(QUOTATION), 'factual', topic.fact)
            add(fmt(POSITIVE)+' '+tailored_tail(), personal)
        else:
            add(BRIDGES[v], 'factual')
            if act in ('late_personal', 'late_mid_sentence'):
                stem = START[v]
                if act == 'late_mid_sentence':
                    stem = 'With those product conditions in mind, '+(stem if stem.startswith('I ') else stem[0].lower()+stem[1:])
                add(stem, 'factual', topic.fact)
                add(topic.product+' for you. '+tailored_tail(), personal)
            elif act == 'late_negative':
                add(fmt(NEGATIVE)+' '+tailored_tail(True), personal)
            else:
                add(fmt(GENERAL), general)
    else:
        raise ValueError(act)

    if act in ('incremental_recommendation', 'incremental_document') and case % 3 == 0:
        # A separate truncated-word checkpoint; it never inherits the completion.
        partial = points[0]['text'][:-2].rstrip()
        points.insert(0, dict(text=partial, profile='fragment', evidence=partial, flags={}))

    # Substantive longer explanations, not greeting/disclaimer padding. In late
    # advice paths the expansion is visible before any recommendation appears.
    if int(stable(topic.key+str(case)+act+'length')[:8], 16) % 4 == 0:
        if act == 'quotation' and case % 2 == 0:
            extension = (f' Another illustrative speaker offers a second opinion about {topic.product}: '
                         f'"{topic.caution}" These two quotations are examples of other speakers\' views; '
                         'I am not adopting either as my recommendation to you.')
            points[-1]['text'] += extension
        elif points[-1]['profile'] not in ('fragment', 'product_fragment', 'product_other', 'other'):
            extension = ' '+(LONG_SECURITY[v % 4] if act == 'security_procedure' else LONG[topic.domain][v % 4])
            at = 2 if act.startswith('late_') else len(points)-1
            insertion_prefix = points[at]['text']
            for point in points[at:]:
                point['text'] = insertion_prefix+extension+point['text'][len(insertion_prefix):]

    # The same follow-up phrases occur after facts, advice and non-advice.
    # Two questions maximum, no requests to repeat a nonexistent earlier turn.
    if case in (2, 8) and act not in ('clarification', 'refusal', 'nonproduct_question'):
        points[-1]['text'] += ' '+fmt(TAILS)
    if case in (7, 10) and act not in ('security_procedure', 'nonproduct_question', 'clarification', 'refusal'):
        points[-1]['text'] += ' '+fmt(DOCUMENT)
        if points[-1]['profile'] == 'product_other':
            points[-1]['profile'] = 'factual'
            points[-1]['evidence'] = fmt(DOCUMENT)
    # Preserve quotation boundaries: punctuation removal could turn a quoted
    # recommendation into an apparent endorsement. No destructive ASR transform.
    surface = int(stable(topic.key+str(case)+act+'voice')[:8], 16) % 4
    if surface == 1:
        for point in points:
            for key in ('text', 'evidence'):
                point[key] = point[key].replace('I am ', "I'm ").replace('I will ', "I'll ")
    for point in points:
        point['frame_id'] = v
        point['basis_type'] = kind
        if point['evidence'] not in point['text']:
            raise ValueError('Authored evidence is absent')
    return clean_repetition(points)
