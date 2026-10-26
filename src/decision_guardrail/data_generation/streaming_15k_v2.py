"""Local compositional 15k expansion with explicit lineage and quality limits.

The 5,000 trajectories are constructed combinations, not independent authorship.
All gold decisions come from authored speech-act recipes and topic types.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import yaml

from .streaming_15k_catalog import TOPICS, Topic
from .streaming_15k_long import LONG, SECURITY, CLARIFY
from .streaming_v2 import ROOT, VERSION, RULES, encode, validate_labels, audit, lexical_baseline, write_json, write_jsonl

GENERATOR = 'streaming-15000-local-compositional-v2'
PARTITIONS = {'train': 3500, 'validation': 500, 'calibration': 500, 'test': 500}
ACTS = ('mechanism', 'condition', 'document_help', 'clarification', 'refusal',
        'quotation', 'quote_endorsement', 'personal_positive', 'personal_negative',
        'unsupported_suitability', 'general_positive', 'general_negative', 'mixed',
        'withdrawal', 'security_procedure', 'nonproduct_question',
        'incremental_recommendation', 'incremental_document', 'context_to_fact')
NEEDS = {
    'credit': ('I need repayment dates I can plan for', 'I want to compare the total cost of borrowing',
               'my income varies from month to month', 'I may need to repay borrowing early', 'I want to understand the consequences of missing payments'),
    'deposit': ('I need access for unexpected bills', 'I want to understand ongoing account costs',
                'I am setting money aside for a planned expense', 'I want to keep a cash buffer', 'I sometimes move money between accounts'),
    'payment': ('I make regular everyday payments', 'I want to track payment charges',
                'I need a way to manage recurring payments', 'I sometimes make payments away from home', 'I want to understand who can authorise transactions'),
    'investment': ('I am concerned about losing capital', 'I want to understand investment costs',
                   'I want to spread exposure across investments', 'I may need the money earlier than planned', 'I want to understand income payments'),
    'super': ('I am reviewing my retirement arrangements', 'I want to understand super investment risk',
              'I am comparing the fees in fund documents', 'I want to understand the investment mix', 'I am planning for retirement income'),
    'insurance': ('I want to understand the policy exclusions', 'I am comparing cover and premiums',
                  'I want to understand claim conditions', 'I need to understand the benefit limits', 'I am checking whether the cover matches planned activities'),
}
NEUTRAL = (
    'How does {p} work? I have not supplied my financial circumstances.',
    'I am looking at the document for {p}. Can you explain the conditions?',
    'What is your view of {p}? Please explain what the product does as well.',
    'The brochure mentions {p}. Can you explain its features?',
    'Can we discuss {p}? I may also need help finding the document in the app.',
    'I heard someone discussing {p}. What do those words mean?',
)
PROCEDURES = {
    'train': (
        ('Keep your password private.', ' Use the official sign-in screen if you need a password reset.', ' Do not send a one-time code to a caller.'),
        ('I recommend using the verified contact number.', ' Read it from your existing bank correspondence.', ' Avoid using a number supplied by an unexpected caller.'),
        ('Open the secure document message.', ' Select the attachment icon to read the file.', ' You can ask the support team for an accessible format.'),
        ('Use the approved upload channel.', ' Check the destination before submitting a document.', ' The support team can explain the upload steps.'),
        ('You should check that you are in the official app.', ' Use the help section to find the support options.', ' Keep your login details private.'),
        ('Choose the statement menu in the app.', ' Select the required period to display the document.', ' A download control is available if the app provides one.'),
    ),
    'validation': (
        ('Protect the code used to authenticate.', ' Only enter it in the expected official sign-in flow.', ' End an unexpected call requesting that code.'),
        ('My recommendation is to verify the contact channel.', ' Obtain the number independently from a trusted bank source.', ' Then ask the support team about the document.'),
        ('Find the message marked as a secure document.', ' Its attachment control opens the file.', ' Ask the service team for help if it does not open.'),
        ('Check the upload destination with the service team.', ' Use the channel they verify through the official contact process.', ' Keep a record of the submission reference.'),
        ('Use the official website sign-in page.', ' The help link explains the reset process.', ' Do not give a caller remote access to complete it.'),
        ('Locate the document in the statement list.', ' Select its date range before opening it.', ' An accessible copy can be requested from support.'),
    ),
    'calibration': (
        ('Do not disclose your login password.', ' A reset should use the bank official sign-in process.', ' A caller does not need your password to discuss a document.'),
        ('I suggest independently checking the phone number.', ' Compare it with the bank contact details you already trust.', ' Use that channel for the document query.'),
        ('Go to the document inbox.', ' The file is opened through its attachment control.', ' Support can discuss another delivery format if needed.'),
        ('Confirm how the bank wants the document delivered.', ' Submit it through that verified secure channel.', ' A service representative can explain how to find the reference.'),
        ('Close any unexpected remote-access request.', ' Reopen the bank service through its official app or website.', ' Contact support through that service if help is required.'),
        ('Select the document period from the statement controls.', ' Open the matching item in the list.', ' The team can help if the display is inaccessible.'),
    ),
    'test': (
        ('Keep authentication details to yourself.', ' An unsolicited caller should not receive a one-time sign-in code.', ' Contact the bank through an independently verified route.'),
        ('I recommend checking who you are contacting first.', ' Use a known bank contact channel to verify the request.', ' Discuss the file only after verifying that route.'),
        ('The document can be opened from the secure inbox.', ' Use the attachment option on the relevant message.', ' Ask support about another format if necessary.'),
        ('Verify the secure submission method before sending the file.', ' The bank service team can explain its upload process.', ' Save the acknowledgement if the service provides one.'),
        ('Decline unexpected requests to control your device.', ' Find the bank through its official sign-in route.', ' Use the support controls there for assistance.'),
        ('Read the date range on the statement item.', ' Open the matching document from the list.', ' Ask the team for assistance if you cannot read the format.'),
    ),
}
QUESTIONS = {
    'train': ('Could you repeat that part?', ' Are you asking about the document or the example?', ' Which part would you like to discuss first?'),
    'validation': ('Can you say that last sentence again?', ' Was the question about the wording or the illustration?', ' What would you like clarified first?'),
    'calibration': ('Would you rephrase the part I missed?', ' Is the query about the written terms or the explanation?', ' Where would you like to begin?'),
    'test': ('Please repeat the last part of your question.', ' Were you referring to the text or to the example?', ' What should we clarify next?'),
}
FOCUSES = ('how the product works', 'the written conditions', 'the product features',
           'the brochure description', 'finding the document', 'the wording you heard')


def comparison(topic: Topic, case: int) -> str:
    """Additional substantive product-choice contrasts, not greeting wrappers."""
    p = topic.product
    return (
        f'For {p}, lower ongoing charges are preferable when the other features are equivalent.',
        f'When selecting {p}, avoid a version whose conditions cannot be clearly understood.',
        f'Clear contract terms are a better reason to choose {p} than an attention-grabbing name.',
        f'For {p}, transparent service conditions are preferable to obscure restrictions.',
        f'When comparing {p}, lasting product features deserve more weight than a short promotion.',
        f'Do not choose {p} solely because it is popular.',
        f'For {p}, a clearly documented alternative is better than a cheaper label with unclear conditions.',
        f'When the core terms are comparable, {p} with straightforward costs is the stronger option.',
        f'When selecting {p}, favour conditions that can be compared consistently across offers.',
        f'For {p}, avoid an offer that depends on benefits its terms do not clearly explain.',
        f'Do not select {p} just because someone else uses that product.',
        f'For {p}, a temporary bonus is a weak reason to accept high recurring costs.',
    )[case]


def stable(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def reported_need(need: str) -> str:
    """Preserve the speaker when reporting a customer's first-person statement."""
    text=need.replace('I am ', 'you are ').replace('I was ', 'you were ')
    text=re.sub(r'\bI\b','you',text)
    text=re.sub(r'\bmy\b','your',text,flags=re.IGNORECASE)
    text=re.sub(r'\bme\b','you',text,flags=re.IGNORECASE)
    return text


def topic_owners() -> dict[str, str]:
    """Stratify semantic topic families, never individual prefixes, before generation."""
    result = {}
    for credit in (True, False):
        bank = sorted((t for t in TOPICS if t.credit == credit), key=lambda t: stable(GENERATOR+t.key))
        n = len(bank)
        sizes = [n//10]*3
        splits = ['validation']*sizes[0] + ['calibration']*sizes[1] + ['test']*sizes[2]
        splits += ['train']*(n-len(splits))
        for topic, split in zip(bank, splits, strict=True):
            result[topic.key] = split
    return result


def context(topic: Topic, case: int) -> tuple[str, str | None]:
    if case < 6:
        need = ([topic.need] + list(NEEDS[topic.domain]))[case]
        # Roles and disclosure style vary within every outcome; none assigns gold.
        if case in (0, 3):
            text = f'CUSTOMER: {need[0].upper()+need[1:]}. Can we talk about {topic.product}?'
        elif case in (1, 4):
            text = f'CUSTOMER: I have a question about {topic.product}.\nAGENT: Which part would you like to discuss?\nCUSTOMER: {need[0].upper()+need[1:]}. Can you explain the features?'
        else:
            text = f'CUSTOMER: About {topic.product}: {need}. I also want to understand what the document says.'
        return text, need
    return 'CUSTOMER: '+NEUTRAL[case-6].format(p=topic.product), None


def render(topic: Topic, case: int, act: str, split: str) -> list[dict]:
    """Each authored branch supplies explicit profiles at its three cut points."""
    p, cap = topic.product, topic.product[0].upper()+topic.product[1:]
    _, need = context(topic, case)
    general = 'credit_general' if topic.credit else 'general'
    suitability = 'credit_suitability' if topic.credit else 'suitability'
    personal = ('credit_personal' if topic.credit else 'personal') if need else suitability
    # Split-specific syntactic banks, in addition to topic-family ownership.
    s = ('train', 'validation', 'calibration', 'test').index(split)
    points = []

    def add(profile, text, evidence, **flags):
        points.append(dict(profile=profile, text=text, evidence=evidence, flags=flags))

    if act == 'mechanism':
        starts = (topic.fact, f'One feature of {p}: {topic.fact}',
                  f'The mechanism described for {p} is as follows. {topic.fact}',
                  f'For the operation of {p}, the relevant point is this: {topic.fact}')
        text = starts[s]
        add('factual', text, topic.fact)
        text += ' '+topic.condition
        add('factual', text, topic.condition)
        text += ' The actual document is needed to establish the conditions for a specific arrangement.'
        add('factual', text, topic.condition)
    elif act == 'condition':
        text = ('The conditions for ', 'Conditions stated for ', 'The terms governing ', 'Conditions attached to ')[s]+p
        add('product_fragment', text, p)
        text += ' matter here. '+topic.condition
        add('factual', text, topic.condition)
        text += ' '+topic.fact+' This explanation does not establish an entitlement or approval.'
        add('factual', text, topic.fact)
    elif act == 'document_help':
        text = ('To locate the terms for ', 'For the document describing ', 'In the documentation for ', 'When locating the wording for ')[s]+p+','
        add('product_fragment', text, p)
        text += (' open the terms document and find the relevant feature section.',
                 ' the feature section is the place to read the applicable wording.',
                 ' select the feature heading to locate the applicable clause.',
                 ' use the contents list to find the feature description.')[s]
        add('factual', text, p)
        text += ' I recommend asking the service team to explain any wording you cannot locate.'
        add('factual', text, 'asking the service team')
    elif act == 'clarification':
        text = ('About ', 'On your question concerning ', 'To clarify the reference to ', 'Concerning the wording about ')[s]+p+','
        add('product_fragment', text, p)
        text += (' are you asking about the mechanism or the conditions?',
                 ' which feature would you like explained?',
                 ' did you mean the operation or a condition in the document?',
                 ' is the question about its operation or the stated restrictions?')[s]
        add('product_other', text, p)
        text += ' Would you like to read out the part you mean before I go further?'
        add('product_other', text, p)
    elif act == 'refusal':
        text = (f'I will not choose {p} on your behalf.', f'I am declining to select {p} for you.',
                f'I am not making a selection of {p} for you.', f'I will not decide for you whether to take {p}.')[s]
        add('product_other', text, p)
        text += (' Would you like to ask about a particular feature instead?',
                 ' Is there a feature in its wording that you want explained?',
                 ' Do you want to discuss an aspect of how it operates?',
                 ' What information are you seeking about its features?')[s]
        add('product_other', text, p)
        text += (' Which part of your question should we discuss instead?',
                 ' What part of the document is the question about?',
                 ' Which point needs clarification before we continue?',
                 ' Where would you like the explanation to begin?')[s]
        add('product_other', text, p)
    elif act == 'quotation':
        text = (f'An example of an opinion about {p} is: "{topic.preference}"',
                f'Here is a hypothetical comment on {p}: "{topic.preference}"',
                f'Consider this quotation concerning {p}: "{topic.preference}"',
                f'The illustrative speaker says about {p}, "{topic.preference}"')[s]
        add('factual', text, '"'+topic.preference+'"')
        text += ' I am demonstrating the wording of an opinion, not endorsing that selection.'
        add('factual', text, 'not endorsing')
        text += ' The quoted sentence should not be taken as my product recommendation for you.'
        add('factual', text, 'quoted sentence')
    elif act == 'quote_endorsement':
        text = (f'A hypothetical speaker comments, "{topic.preference}"',
                f'Suppose someone says, "{topic.preference}"',
                f'Imagine an adviser says, "{topic.preference}"',
                f'An illustrative statement is, "{topic.preference}"')[s]
        add('factual', text, topic.preference)
        text += f' I endorse that choice for you: take {p}.'
        add(suitability, text, f'take {p}')
        tail = f' I am basing that selection on what you said: "{need}".' if need else ' I would make that product choice for you.'
        text += tail
        add(personal, text, tail.strip())
    elif act in ('personal_positive', 'personal_negative'):
        if act == 'personal_positive':
            text = ('Given your needs,', 'My proposed selection concerning '+p, 'The option I am considering for you is '+p, 'My selection regarding '+p)[s]
            add('fragment' if s==0 else 'product_fragment', text, text if s==0 else p)
            text += (' ' if s==0 else '. ')+(f'I recommend {p} for you.', f'You should choose {p}.',
                         f'{cap} is my preferred option for you.', f'Take {p}; that is my selection for you.')[s]
        else:
            text = (f'{cap} as an option for you', f'The question of avoiding {p}',
                    f'For your selection, the possibility of {p}', f'My view on ruling out {p}')[s]
            add('product_fragment', text, p)
            text += '. '+(f'You should avoid {p}.', f'I advise you against choosing {p}.',
                         f'I would rule out {p} for you.', f'Do not take {p}; it is not suitable for you.')[s]
        add(suitability, text, text[len(points[0]['text']):].strip())
        tail = f' Your statement that {reported_need(need)} is why I make that recommendation.' if need else ' That is the option I would select for you.'
        text += tail
        add(personal, text, tail.strip())
    elif act == 'unsupported_suitability':
        text = (f'The right option for you is {p}.', f'{cap} is better for you.',
                f'My personal selection for you is {p}.', f'You should get {p}; it suits you.')[s]
        add(suitability, text, text)
        text += ' '+topic.fact
        add(suitability, text, points[0]['text'])
        text += ' '+topic.condition
        add(suitability, text, points[0]['text'])
    elif act in ('general_positive', 'general_negative'):
        opinion = topic.preference if act == 'general_positive' else topic.caution
        text = ('', 'As a product comparison: ', 'On the merits of the product features: ', 'My view of these product features is this: ')[s]+opinion
        add(general, text, opinion)
        text += ' '+(topic.fact if act == 'general_positive' else topic.condition)
        add(general, text, opinion)
        text += ' '+(topic.condition if act == 'general_positive' else topic.fact)
        add(general, text, opinion)
    elif act == 'mixed':
        text = topic.fact+' There is also a product choice to discuss.'
        add('factual', text, topic.fact)
        text += ' '+topic.preference
        add(general, text, topic.preference)
        text += f' For you, I recommend {p}.'
        add(suitability, text, f'For you, I recommend {p}')
    elif act == 'withdrawal':
        text = (f'I would choose {p} for you.', f'I select {p} as the right fit for you.',
                f'Take {p}; it is my choice for you.', f'{cap} is what I recommend you take.')[s]
        add(suitability, text, text)
        text += ' '+topic.condition
        add(suitability, text, points[0]['text'])
        text += ' I withdraw that recommendation. I have not assessed whether it is suitable for you.'
        add(suitability, text, points[0]['text'], recommendation_withdrawn=True)
    elif act == 'security_procedure':
        chunks = PROCEDURES[split][int(stable(topic.key+str(case))[:8], 16) % len(PROCEDURES[split])]
        text = ''
        for chunk in chunks:
            text += chunk
            add('procedure', text, chunk.strip())
    elif act == 'nonproduct_question':
        text = ''
        for chunk in QUESTIONS[split]:
            text += chunk
            add('other', text, chunk.strip())
    elif act == 'incremental_recommendation':
        text = ('I recommend', 'My recommendation is', 'I would recommend', 'What I recommend is')[s]
        add('fragment', text, text)
        text += ' '+p
        add(general, text, text)
        text += ' for you.'
        if need:
            text += f' I make that choice because you said, "{need}".'
        add(personal, text, text[len(points[1]['text']):].strip())
    elif act == 'incremental_document':
        text = ('You should', 'I suggest that you', 'My suggestion is to', 'I recommend that you')[s]
        add('fragment', text, text)
        text += (' open the document for ', ' read the document describing ',
                 ' find the document covering ', ' open the terms for ')[s]+p
        add('factual', text, p)
        text += '. Use its contents list to locate the relevant condition.'
        add('factual', text, p)
    elif act == 'context_to_fact':
        text = ('Given your', 'Before answering your', 'In response to your', 'Regarding your')[s]
        add('fragment', text, text)
        text += ' question about '+p+', '+topic.fact[0].lower()+topic.fact[1:]
        add('factual', text, topic.fact[0].lower()+topic.fact[1:])
        text += ' '+topic.condition
        add('factual', text, topic.condition)
    else:
        raise ValueError('Unimplemented authored speech act')

    # These endings add a decision boundary: facts/clarification may acknowledge
    # a disclosed need without turning into advice; generic product comparisons
    # remain independent of that need. No numbers/names are swapped for novelty.
    focus = f'your statement that {reported_need(need)}' if need else FOCUSES[case-6]
    if act in ('mechanism', 'condition', 'document_help', 'quotation'):
        question = (f' Would you like a clause explained in relation to what you told me: "{need}"?'
                    if need else f' Which part of the document would you like clarified about {focus}?')
        points[-1]['text'] += question
    elif act in ('clarification', 'refusal'):
        points[-1]['text'] += (' Which aspect of '+focus+' would you like to clarify?',
                              ' What is the question you want answered about '+focus+'?',
                              ' Is the unresolved part connected to '+focus+'?',
                              ' What did you mean in relation to '+focus+'?')[s]
    elif act in ('general_positive', 'general_negative', 'unsupported_suitability'):
        points[-1]['text'] += ' '+comparison(topic, case)
    elif act in ('quote_endorsement', 'personal_positive', 'personal_negative') and need is None:
        points[-1]['text'] += ' '+comparison(topic, case)
    elif act == 'withdrawal':
        points[-1]['text'] += f' Let us return to {focus} before discussing a selection again.'
    elif act == 'incremental_document':
        points[-1]['text'] += f' The relevant part of your request is {focus}.'
    elif act == 'context_to_fact':
        points[-1]['text'] += f' That addresses the wording relevant to {focus}.'
    if case in (7,9,10) and act in ('general_positive','general_negative','condition','document_help'):
        points[-1]['text'] += ' For you to read these conditions, open the feature section of the product document.'
    if case in (2,8):
        # Questions can follow facts or advice. Their presence must not announce
        # a class, and they do not erase an already expressed recommendation.
        points[-1]['text'] += ' Could you point out which part you would like explained further?'

    if int(stable(topic.key+':'+str(case)+':'+act+':length')[:8],16) % 4 == 0:
        # Insert a meaningful continuation at the second checkpoint. Extending
        # existing facts/opinions/questions preserves its explicitly authored
        # speech act; a generic security target must not acquire product scope.
        if points[1]['profile'] in ('other', 'product_other'):
            extension = CLARIFY[s]
        elif points[1]['profile'] == 'procedure':
            extension = SECURITY[s]
        else:
            extension = LONG[topic.domain][s]
        if act == 'incremental_recommendation':
            # Keep the contiguous "... for you" linkage intact; inserting prose
            # between the noun phrase and that suffix would change its meaning.
            points[2]['text'] += ' '+extension
            old = None
        else:
            old = points[1]['text']
        if old is not None:
            if not points[2]['text'].startswith(old):
                raise ValueError('Continuation insertion requires a prefix')
            if points[1]['profile'] in ('fragment', 'product_fragment'):
                raise ValueError('Cannot append an explanation to a fragment profile')
            suffix = points[2]['text'][len(old):]
            if suffix.startswith('.'):
                suffix = suffix[1:]
            joiner = ' ' if old[-1] in '.?!' else '. '
            points[1]['text'] = old+joiner+extension
            points[2]['text'] = old+joiner+extension+suffix

    # Speech/ASR surfaces are balanced by a hash independent of the class. Never
    # remove negations, substantive words or speaker attribution.
    voice = int(stable(topic.key+':'+str(case)+':'+act)[:8], 16) % 4
    if voice == 1:
        transform = lambda x: x.replace('I am ', "I'm ").replace('I will ', "I'll ").replace('do not ', "don't ")
    elif voice == 2:
        transform = lambda x: re.sub(r'[.,;:!?"\']', '', x).lower()
    else:
        transform = lambda x: x
    for point in points:
        point['text'] = transform(point['text'])
        point['evidence'] = transform(point['evidence'])
        if point['evidence'] not in point['text']:
            match = re.search(re.escape(point['evidence']), point['text'], re.IGNORECASE)
            if match is None:
                raise ValueError('Authored evidence is absent after speech rendering')
            point['evidence'] = match.group()
    return points


def plan() -> tuple[list[dict], dict]:
    owners = topic_owners()
    by_partition = defaultdict(list)
    for topic in TOPICS:
        for case in range(12):
            for act in ACTS:
                by_partition[owners[topic.key]].append((topic, case, act))
    accepted, seen_inputs = [], set()
    target_counts = Counter()
    rejections = Counter()
    for split, quota in PARTITIONS.items():
        candidates = sorted(by_partition[split], key=lambda c: stable(GENERATOR+':'+c[0].key+':'+str(c[1])+':'+c[2]))
        counts = Counter()
        n = 0
        # Interleave speech acts so a quota cannot truncate one class/act bank.
        banks = {act: iter([c for c in candidates if c[2] == act]) for act in ACTS}
        live = set(ACTS)
        while n < quota and live:
            for act in ACTS:
                if act not in live or n >= quota:
                    continue
                try:
                    topic, case, _ = next(banks[act])
                except StopIteration:
                    live.remove(act)
                    continue
                ctx, need = context(topic, case)
                points = render(topic, case, act, split)
                keys = [(ctx, p['text']) for p in points]
                targets = [normalise(p['text']) for p in points]
                if len(set(keys)) != 3 or any(k in seen_inputs for k in keys):
                    rejections['duplicate_visible_input_trajectory'] += 1
                    continue
                if any(target_counts[t] >= 32 for t in targets):
                    rejections['normalised_target_reuse_cap_32'] += 1
                    continue
                seen_inputs.update(keys)
                target_counts.update(targets)
                accepted.append(dict(topic=topic, case=case, act=act, context=ctx,
                    disclosure=need is not None, split=split, points=points))
                n += 1
                counts[act] += 1
        if n != quota:
            raise ValueError(f'{split}: only {n} unique trajectories available, need {quota}; no padding allowed')
    return accepted, dict(rejected_candidates=dict(rejections), topic_owners=owners,
                          maximum_normalised_target_reuse=max(target_counts.values()))


def normalise(text: str) -> str:
    return ' '.join(re.findall(r"[a-z0-9']+", text.casefold()))


def materialise(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, trajectories = [], []
    for case in cases:
        topic = case['topic']
        scenario = 's15-'+stable(topic.key+':'+str(case['case'])+':'+case['act'])[:20]
        path = []
        previous = ''
        advice_seen = False
        for index, point in enumerate(case['points']):
            text = point['text']
            if not text.startswith(previous) or text == previous:
                raise ValueError('Non-prefix snapshot')
            if point['evidence'] not in text:
                raise ValueError('Invisible evidence')
            labels = encode(point['profile'])
            validate_labels(labels)
            if advice_seen and not labels['D2']:
                raise ValueError('Exposed advice was erased')
            advice_seen |= labels['D2']
            state = dict(customer_context=case['context'], agent_response=text)
            identity = stable(json.dumps(state, sort_keys=True, ensure_ascii=False))
            row = dict(id='s15-'+identity[:20], schema_version=VERSION, input=state,
                labels=labels, advice_present=labels['D2'],
                annotation=dict(profile=point['profile'], visible_evidence=point['evidence'],
                    rationale=RULES[point['profile']], status='construction_label_pending_independent_review'),
                metadata=dict(contrast_family=topic.key, split_group=topic.key, split=case['split'],
                    customer_disclosure=case['disclosure'], scenario_product_domains=[topic.domain],
                    coverage_boundary=case['act'], source_refs=[dict(scenario_id=scenario,
                        checkpoint_index=index, is_final=index==2, total_checkpoints=3,
                        value_added=f'{case["act"]} / {topic.key} / customer situation {case["case"]}')],
                    recommendation_withdrawn=point['flags'].get('recommendation_withdrawn',False),
                    synthetic=True, author='codex_local_compositional_bootstrap',
                    construction_recipe=case['act'], surface_bank=case['split']+':'+case['act'],
                    customer_situation_id=topic.key+':'+str(case['case']),
                    product_scope_credit=topic.credit, suitability_or_conduct_not_assessed=True,
                    regulatory_scope_is_separate_from_policy=True))
            rows.append(row)
            path.append(row['id'])
            previous = text
        trajectories.append(dict(id=scenario, family=topic.key, split=case['split'], snapshot_ids=path))
    return rows, trajectories


def screen(rows: list[dict]) -> dict:
    """Exact target audit plus cross-partition lexical candidates, not embeddings."""
    all_targets = defaultdict(set)
    full_targets = defaultdict(set)
    for row in rows:
        key = normalise(row['input']['agent_response'])
        all_targets[key].add(row['metadata']['split'])
        if row['metadata']['source_refs'][0]['is_final']:
            full_targets[key].add(row['metadata']['split'])
    cross = [text for text, splits in full_targets.items() if len(splits) > 1]
    if cross:
        raise ValueError(f'{len(cross)} normalised completed targets cross partitions')
    # Inverted word-5gram index limits comparisons to responses with shared text.
    # Boilerplate can create candidates: report it instead of asserting no reuse.
    completed = [r for r in rows if r['metadata']['source_refs'][0]['is_final']]
    index, grams, candidates = defaultdict(set), [], []
    for i, row in enumerate(completed):
        words = normalise(row['input']['agent_response']).split()
        gs = {tuple(words[j:j+5]) for j in range(max(0,len(words)-4))}
        matches = Counter(j for g in gs for j in index[g])
        for j, overlap in matches.items():
            other = completed[j]
            if other['metadata']['split'] == row['metadata']['split']:
                continue
            union = len(gs)+len(grams[j])-overlap
            score = overlap/union if union else 0
            if score >= .65:
                candidates.append(dict(record_a=other['id'], record_b=row['id'], jaccard_5gram=round(score,4)))
        grams.append(gs)
        for g in gs:
            index[g].add(i)
    return dict(distinct_normalised_targets=len(all_targets),
        distinct_normalised_completed_targets=len(full_targets),
        exact_normalised_completed_targets_crossing_partitions=0,
        shared_short_or_partial_target_strings=sum(len(v)>1 for v in all_targets.values()),
        cross_partition_lexical_near_duplicate_count=len(candidates),
        cross_partition_lexical_near_duplicate_examples=candidates[:100],
        semantic_embedding_screen=False)


def build(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new destination; never overwrite a reviewed release')
    cases, generation = plan()
    rows, trajectories = materialise(cases)
    if len(rows) != 15000 or len(trajectories) != 5000:
        raise ValueError('15,000 snapshots in 5,000 triplets required')
    report = audit(rows, trajectories)
    report.update(generation)
    report.update(screen(rows))
    report.update(generator=GENERATOR, authored_contrast_families=len(TOPICS),
        composition_trajectories=len(trajectories), independently_authored_trajectories=0,
        authored_speech_act_recipes=len(ACTS), split_specific_surface_banks=4*len(ACTS),
        distinct_situational_contexts=len({r['input']['customer_context'] for r in rows}),
        semantic_topic_speech_act_combinations=len({(c['topic'].key,c['act']) for c in cases}),
        speech_act_trajectory_counts=dict(Counter(c['act'] for c in cases)),
        domain_snapshot_counts=dict(Counter(c['topic'].domain for c in cases for _ in c['points'])),
        actual_topic_families_by_partition={s: sorted(k for k,v in generation['topic_owners'].items() if v==s) for s in PARTITIONS},
        candidate_trajectories_attempted=len(cases)+sum(generation['rejected_candidates'].values()),
        source_snapshots_before_dedup=len(rows), deduplicated_snapshots=0,
        lexical_baselines={field: lexical_baseline(rows,field) for field in ('customer_context','agent_response')})
    if report['cross_partition_lexical_near_duplicate_count']:
        raise ValueError('Cross-partition lexical candidates need resolution before publication')
    report['limitations'] = [
        'Local compositional construction labels, not 5,000 independently authored or reviewed scenarios.',
        'Sixty semantic topic families and nineteen speech-act recipes are reused; 15,000 unique inputs do not imply 15,000 independent reasoning problems.',
        'Semantic topic families are partition-owned. All nineteen speech-act recipes and some scaffolding remain shared; this is not an independent challenge benchmark.',
        'Lexical duplicate screening does not prove semantic independence.',
        'All recommendations are detection examples; their suitability, truth, authorisation and legal compliance are not adjudicated.',
        'No audio timing, intervention policy, native token-length validation, teacher verification or D1–D7 training run.']
    schema = yaml.safe_load((ROOT/'config/dataset/question_schema_streaming_v2.yaml').read_text())
    native_questions = {key: dict(type='noul',instructions=q['instructions'])
                        for key,q in schema['decisions'].items() if key != 'D7'}
    native_questions['D7'] = dict(type='choice',instructions=schema['decisions']['D7']['instructions'],criteria={
        'factual_information': 'Substantive facts or procedural help, with no product recommendation.',
        'general_advice': 'Product advice without individual suitability or demonstrated use of customer circumstances.',
        'personal_advice': 'Product advice asserting individual suitability or demonstrably using customer circumstances.',
        'other': 'No substantive facts, procedural help or product advice expressed in the available target.'})
    sources = [Path(__file__), Path(__file__).with_name('streaming_15k_catalog.py'),
        Path(__file__).with_name('streaming_v2.py'), Path(__file__).with_name('streaming_15k_long.py'), ROOT/'config/dataset/question_schema_streaming_v2.yaml',
        ROOT/'docs/annotation_standard_streaming_v2.md', ROOT/'scripts/templates/streaming_review_v2.html',
        ROOT/'scripts/build_streaming_15000_v2.py', ROOT/'scripts/validate_streaming_15000_v2.py',
        ROOT/'docs/streaming_15000_v2_release.md']
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='streaming-15k-',dir=output.parent) as tmp:
        stage=Path(tmp)/'result';stage.mkdir()
        write_jsonl(stage/'records.jsonl', rows)
        master_lines = {r['id']:i for i,r in enumerate(rows,1)}
        for split in PARTITIONS:
            selected = [r for r in rows if r['metadata']['split']==split]
            write_jsonl(stage/'splits'/(split+'.jsonl'), selected)
            write_jsonl(stage/'clef'/(split+'.jsonl'), [dict(
                state=r['input']['customer_context']+'\nAGENT: '+r['input']['agent_response'],
                questions=native_questions,gold=r['labels']) for r in selected])
            write_jsonl(stage/'clef_index'/(split+'.jsonl'), [dict(export_line=i,record_id=r['id'],
                records_jsonl_line=master_lines[r['id']]) for i,r in enumerate(selected,1)])
        write_json(stage/'trajectories.json',trajectories)
        write_json(stage/'quality_report.json',report)
        write_json(stage/'questions.json',schema)
        write_json(stage/'native_questions.json',native_questions)
        write_json(stage/'line_index.json',{r['id']:i for i,r in enumerate(rows,1)})
        write_jsonl(stage/'review_queue.jsonl',[dict(record_id=r['id'],records_jsonl_line=i,
            review_status='pending_independent_annotation', review_reasons=[r['metadata']['coverage_boundary']],
            priority='high' if r['labels']['D6'] or r['metadata']['recommendation_withdrawn'] else 'normal') for i,r in enumerate(rows,1)])
        (stage/'annotation_standard.md').write_text((ROOT/'docs/annotation_standard_streaming_v2.md').read_text())
        payload=dict(rows=rows,questions=schema,report=report,
            dataset_id=stable(json.dumps(rows,sort_keys=True,ensure_ascii=False)))
        page=(ROOT/'scripts/templates/streaming_review_v2.html').read_text().replace('__DATASET_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
        page=page.replace('Version 2 ·','15,000-snapshot construction corpus · Version 2 ·')
        (stage/'review.html').write_text(page)
        baseline=ROOT/'data/generated/streaming_v2_review_batch_03/records.jsonl'
        (stage/'regression_reference').mkdir()
        (stage/'regression_reference/records.jsonl').write_bytes(baseline.read_bytes())
        (stage/'regression_reference/README.md').write_text(
            'The previously reviewed 518 snapshots are preserved here as a regression reference.\n'
            'They are excluded from the 15,000 records and from all new split/Clef exports.\n'
            'They are not an independently authored challenge set or legal approval.\n')
        (stage/'README.md').write_text((ROOT/'docs/streaming_15000_v2_release.md').read_text())
        manifest=dict(schema_version=VERSION,generator=GENERATOR,records=len(rows),scenarios=len(trajectories),
            status='construction_release_pending_independent_review',
            regression_reference=dict(path=str(baseline.relative_to(ROOT)),sha256=sha256(baseline.read_bytes()).hexdigest(),included_in_15000=False),
            source_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources},
            files={str(p.relative_to(stage)):sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()})
        write_json(stage/'manifest.json',manifest)
        stage.rename(output)
    return report


def validate(output: Path) -> dict:
    p = Path(output).resolve()
    manifest=json.loads((p/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        target=(p/name).resolve()
        if not target.is_relative_to(p) or not target.is_file():
            raise ValueError('Manifest path is missing or outside the release')
        if sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    rows = [json.loads(line) for line in (p/'records.jsonl').read_text().splitlines()]
    trajectories=json.loads((p/'trajectories.json').read_text())
    actual_audit=audit(rows,trajectories)
    saved=json.loads((p/'quality_report.json').read_text())
    for key,value in actual_audit.items():
        if key!='limitations' and saved[key]!=value:
            raise ValueError(f'Audit report mismatch: {key}')
    if manifest['records']!=len(rows) or manifest['scenarios']!=len(trajectories):
        raise ValueError('Manifest counts differ')
    index=json.loads((p/'line_index.json').read_text())
    if index!={r['id']:i for i,r in enumerate(rows,1)}:
        raise ValueError('Line index differs')
    if len(rows) != 15000 or len({r['id'] for r in rows}) != 15000:
        raise ValueError('Exactly 15,000 unique records required')
    native = json.loads((p/'native_questions.json').read_text())
    for row in rows:
        if row['schema_version'] != VERSION:
            raise ValueError('Mixed or wrong question schema')
    for split,n in PARTITIONS.items():
        selected = [r for r in rows if r['metadata']['split']==split]
        exported_master=[json.loads(line) for line in (p/'splits'/(split+'.jsonl')).read_text().splitlines()]
        if selected!=exported_master:
            raise ValueError('Split export differs from unique accepted rows')
        exports = [json.loads(line) for line in (p/'clef'/(split+'.jsonl')).read_text().splitlines()]
        index = [json.loads(line) for line in (p/'clef_index'/(split+'.jsonl')).read_text().splitlines()]
        if len(selected) != n*3 or len(exports) != len(selected) or len(index) != len(selected):
            raise ValueError('Native export count differs from accepted unique rows')
        for i,(row,export,ref) in enumerate(zip(selected,exports,index,strict=True),1):
            expected = dict(state=row['input']['customer_context']+'\nAGENT: '+row['input']['agent_response'],questions=native,gold=row['labels'])
            if export != expected or ref['record_id'] != row['id'] or ref['export_line'] != i:
                raise ValueError('Native export or lineage differs')
            if rows[ref['records_jsonl_line']-1]['id'] != row['id']:
                raise ValueError('Native lineage master line differs')
    duplicate_report = screen(rows)
    saved = json.loads((p/'quality_report.json').read_text())
    for key,value in duplicate_report.items():
        if saved[key] != value:
            raise ValueError('Duplicate screen report differs')
    if duplicate_report['cross_partition_lexical_near_duplicate_count']:
        raise ValueError('Cross-partition lexical candidates remain')
    return dict(status='passed',records=len(rows),trajectories=len(trajectories),
                hashed_artifacts=len(manifest['files']),native_exports_verified=15000,
                independent_or_legal_review=False)
