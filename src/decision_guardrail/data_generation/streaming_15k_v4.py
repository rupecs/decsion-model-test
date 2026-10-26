"""Versioned v4 construction/export pipeline; prior releases are immutable."""
from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
import yaml
from .streaming_15k_catalog import TOPICS, Topic
from .streaming_15k_v4_content import repeated_sentences, BRIDGES, ACTS, FRAME_POOLS, basis, context, frame_id, render
from .streaming_v2 import ROOT, RULES, encode, validate_labels, audit as base_audit, lexical_baseline, write_json, write_jsonl
from .streaming_15k_v2 import stable

VERSION = 'banking_streaming_d1_d7_v4_binary_draft'
GENERATOR = 'streaming-15000-local-compositional-v4'
PARTITIONS = {'train': 10500, 'validation': 1500, 'calibration': 1500, 'test': 1500}


def topic_owners() -> dict[str, str]:
    owners = {t.key: 'train' for t in TOPICS}
    banks = {domain: sorted([t for t in TOPICS if t.domain == domain],
                             key=lambda t: stable(GENERATOR+t.key))
             for domain in sorted({t.domain for t in TOPICS})}
    # Four credit topics, one from each of the five other domains, plus one
    # additional deposit topic per held-out set: 10 distinct families each.
    for split in ('validation', 'calibration', 'test'):
        selected = [banks['credit'].pop() for _ in range(4)]
        selected += [banks[d].pop() for d in banks if d != 'credit']
        selected += [banks['deposit'].pop()]
        for topic in selected:
            owners[topic.key] = split
    return owners


def plan() -> tuple[list[dict], dict]:
    owners = topic_owners()
    accepted, seen_inputs = [], set()
    target_counts, rejections = Counter(), Counter()
    final_grams, final_index, quarantined = [], defaultdict(set), []
    for split, quota in PARTITIONS.items():
        candidates = sorted([(t, case, act) for t in TOPICS if owners[t.key] == split
                             for case in range(18) for act in ACTS],
                            key=lambda c: stable(GENERATOR+c[0].key+str(c[1])+c[2]))
        n = 0
        def accept(topic, case, act):
            nonlocal n
            points = render(topic, case, act, split)
            if not points:
                rejections['inapplicable_or_redundant_candidate'] += 1
                return False
            left = quota-n-len(points)
            if left < 0 or left in (1, 2):
                rejections['whole_trajectory_snapshot_quota'] += 1
                return False
            ctx, need = context(topic, case)
            keys = [(normalise(ctx), normalise(p['text'])) for p in points]
            targets = [normalise(p['text']) for p in points]
            if len(set(keys)) != len(points) or any(k in seen_inputs for k in keys):
                rejections['duplicate_visible_input_trajectory'] += 1
                return False
            if any(target_counts[t]+targets.count(t) > 32 for t in targets if len(t.split()) >= 8):
                rejections['normalised_target_reuse_cap_32'] += 1
                return False
            words = targets[-1].split()
            grams = {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}
            matches = Counter(j for gram in grams for j in final_index[gram])
            for j, overlap in matches.items():
                other_split, other_topic, other_grams = final_grams[j]
                union = len(grams)+len(other_grams)-overlap
                if other_split != split and union and overlap/union >= .65:
                    rejections['cross_partition_lexical_candidate_quarantined_before_export'] += 1
                    quarantined.append(dict(split=split, topic=topic.key, case=case, act=act,
                        candidate_sha256=stable(ctx+'\n'+points[-1]['text']),
                        matched_split=other_split, matched_topic=other_topic,
                        jaccard_5gram=round(overlap/union,4), included_in_exports=False))
                    return False
            seen_inputs.update(keys)
            target_counts.update(targets)
            for gram in grams:
                final_index[gram].add(len(final_grams))
            final_grams.append((split, topic.key, grams))
            accepted.append(dict(topic=topic, case=case, act=act, context=ctx,
                                 disclosure=need is not None, split=split, points=points))
            n += len(points)
            return True
        # Reserve actual valid examples for every owned frame before filling the
        # quota. Quota truncation cannot silently remove a rare frame/need case.
        for act in ACTS:
            for frame in FRAME_POOLS[split]:
                if not any(accept(t, case, act) for t,case,a in candidates
                           if a == act and frame_id(t,case,act,split) == frame):
                    raise ValueError(f'No valid frame example: {split}:{act}:{frame}')
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
                accept(topic, case, act)
        if n != quota:
            raise ValueError(f'{split}: {n} snapshots available, require {quota}; no padding')
    return accepted, dict(rejected_candidates=dict(rejections), topic_owners=owners,
                          lexical_candidates_excluded_before_all_exports=quarantined,
                          maximum_normalised_target_reuse=max(target_counts.values()),
                          maximum_substantive_target_reuse=max(n for t,n in target_counts.items() if len(t.split()) >= 8))


def audit(rows, trajectories):
    report = base_audit(rows, trajectories)
    report['schema_version'] = VERSION
    report['repeated_normalised_targets'] = report['repeated_normalised_targets'][:100]
    # The frozen helper uses substrings ("for your" matches "for you"). V3
    # diagnostics use word boundaries and do not inherit that false coverage.
    for phrase in report['phrase_screen']:
        matches = [r for r in rows if re.search(r'\b'+re.escape(phrase)+r'\b',
                   r['input']['agent_response'], re.IGNORECASE)]
        counts = Counter(r['labels']['D7'] for r in matches)
        report['phrase_screen'][phrase] = dict(n=len(matches), by_class=dict(counts),
            majority_precision=max(counts.values())/len(matches) if matches else None)
    return report


def acceptance(rows, trajectories):
    """Dataset review checks, not independent annotation or model evaluation."""
    by_id = {r['id']: r for r in rows}
    sequences = [[by_id[k] for k in t['snapshot_ids']] for t in trajectories]
    advice = [s for s in sequences if any(r['advice_present'] for r in s)]
    late = [s for s in advice if s[-1]['advice_present'] and
            not any(r['advice_present'] for r in s[:-1])]
    frames = defaultdict(set)
    normalized_inputs = set()
    for row in rows:
        m = row['metadata']
        key = (normalise(row['input']['customer_context']), normalise(row['input']['agent_response']))
        if key in normalized_inputs:
            raise ValueError('Normalized visible input duplicate')
        normalized_inputs.add(key)
        if repeated_sentences(row['input']['agent_response']):
            raise ValueError('Repeated complete sentence within a target')
        frames[m['split']+':'+m['construction_recipe']].add(m['frame_id'])
        if m['frame_id'] not in FRAME_POOLS[m['split']]:
            raise ValueError('Frame leakage')
        if row['labels']['D3'] and m['customer_basis_type'] not in ('objective_or_need', 'financial_situation'):
            raise ValueError('D3 requires an authored financial circumstance/need')
        if row['labels']['D3']:
            needed = m['customer_basis_text']
            if not needed or not any(s in row['input']['agent_response'] for s in
                                     (needed, needed.replace('I am ', "I'm "))):
                raise ValueError('D3 basis is not visible in the target')
        if row['input']['agent_response'].count('?') > 2:
            raise ValueError('Too many clarification questions')
        if m['construction_recipe'] == 'security_procedure' and 'signing in securely' not in row['input']['customer_context']:
            raise ValueError('Security answer without matching request')
        if m['construction_recipe'] == 'personal_negative' and 'the option I would select for you' in row['input']['agent_response']:
            raise ValueError('Contradictory negative tail')
    for seq in sequences:
        if len({r['input']['customer_context'] for r in seq}) != 1:
            raise ValueError('Context changed inside a trajectory')
        for i, row in enumerate(seq):
            ref = row['metadata']['source_refs'][0]
            if ref['checkpoint_index'] != i or ref['total_checkpoints'] != len(seq) or ref['is_final'] != (i == len(seq)-1):
                raise ValueError('Checkpoint lineage differs from trajectory')
            if i and (not row['input']['agent_response'].startswith(seq[i-1]['input']['agent_response']) or
                      row['input']['agent_response'] == seq[i-1]['input']['agent_response']):
                raise ValueError('Target does not strictly extend previous snapshot')
            if i and seq[i-1]['advice_present'] and not row['advice_present']:
                raise ValueError('Advice exposure erased')
            if row['metadata']['recommendation_withdrawn_in_target'] and not row['advice_present']:
                raise ValueError('Withdrawal without advice exposure')
    if len(late)/len(advice) < .15:
        raise ValueError('Insufficient late-onset advice trajectories')
    if sum(len(s) > 3 for s in sequences) < 500:
        raise ValueError('Insufficient longer trajectories')
    for split, pool in FRAME_POOLS.items():
        topics = {r['metadata']['contrast_family'] for r in rows if r['metadata']['split'] == split}
        if len(topics) != (30 if split == 'train' else 10):
            raise ValueError('Topic ownership coverage differs')
        for act in ACTS:
            if frames[split+':'+act] != set(pool):
                raise ValueError(f'Frame coverage missing: {split}:{act}')
    length_outcomes = {}
    length_classes = {}
    for split in PARTITIONS:
        for length in (5, 6):
            seqs = [s for s in sequences if len(s)==length and s[0]['metadata']['split']==split]
            counts = Counter('advice' if s[-1]['advice_present'] else 'no_advice' for s in seqs)
            if min(counts.get('advice',0), counts.get('no_advice',0)) < 10:
                raise ValueError('Long trajectory outcome coverage is missing')
            fraction = counts['advice']/len(seqs)
            if not .2 <= fraction <= .8:
                raise ValueError('Long trajectory length is too predictive of advice')
            length_outcomes[f'{split}:{length}'] = dict(counts)
            classes = Counter(s[-1]['labels']['D7'] for s in seqs)
            if set(classes) != {'other','factual_information','general_advice','personal_advice'}:
                raise ValueError('Long trajectory length lacks a final class')
            length_classes[f'{split}:{length}'] = dict(classes)
    bridges = {}
    for i, bridge in enumerate(BRIDGES):
        exposed = [s for s in sequences if bridge in s[-1]['input']['agent_response']]
        counts = Counter('advice' if s[-1]['advice_present'] else 'no_advice' for s in exposed)
        if min(counts.get('advice',0), counts.get('no_advice',0)) < 2:
            raise ValueError('Bridge predicts future advice without counterexamples')
        bridges[str(i)] = dict(phrase=bridge, final_outcomes=dict(counts))
    d3_counts, d3_contexts = {}, {}
    for split in PARTITIONS:
        selected = [r for r in rows if r['metadata']['split']==split and r['labels']['D3']]
        d3_counts[split] = len(selected)
        d3_contexts[split] = len({r['input']['customer_context'] for r in selected})
        if len(selected) < (1200 if split=='train' else 150) or d3_contexts[split] < (200 if split=='train' else 40):
            raise ValueError('Insufficient D3 snapshots or distinct financial contexts')
    brief = 'brief explanation or a detailed one'
    brief_classes = {r['labels']['D7'] for r in rows if brief in r['input']['agent_response']}
    if brief_classes != {'other', 'factual_information', 'general_advice', 'personal_advice'}:
        raise ValueError('Explanation-length question has become a class cue')
    personal = [r for r in rows if r['labels']['D7']=='personal_advice']
    d4_only = sum(r['labels']['D4'] and not r['labels']['D3'] for r in personal)
    # Export rather than hide predictive phrases. Meaningful advice wording is
    # expected to correlate with advice; precision is not an annotation rule.
    grams = defaultdict(Counter)
    for row in rows:
        words = normalise(row['input']['agent_response']).split()
        for n in (2, 3, 4):
            for g in set(tuple(words[i:i+n]) for i in range(len(words)-n+1)):
                grams[' '.join(g)][row['labels']['D7']] += 1
    cues = [dict(phrase=g, support=sum(c.values()), by_class=dict(c),
                 majority_precision=max(c.values())/sum(c.values()))
            for g,c in grams.items() if sum(c.values()) >= 100 and max(c.values())/sum(c.values()) > .95]
    cues.sort(key=lambda c: (-c['support'], c['phrase']))
    return dict(acceptance_gates_passed=True,
                advice_trajectories=len(advice), final_checkpoint_first_advice_trajectories=len(late),
                final_checkpoint_first_advice_fraction=len(late)/len(advice),
                trajectories_over_three_checkpoints=sum(len(s)>3 for s in sequences),
                checkpoint_count_distribution=dict(Counter(str(len(s)) for s in sequences)),
                long_trajectory_final_outcomes_by_split_and_length=length_outcomes,
                long_trajectory_final_classes_by_split_and_length=length_classes,
                bridge_final_outcome_counts=bridges,
                repeated_sentence_targets=0,
                D3_positive_snapshots_by_split=d3_counts,
                D3_distinct_customer_contexts_by_split=d3_contexts,
                D3_positive_trajectories=sum(any(r['labels']['D3'] for r in s) for s in sequences),
                personal_advice_D4_only_snapshots=d4_only,
                personal_advice_D4_only_fraction=d4_only/len(personal),
                observed_frame_ids={k:sorted(v) for k,v in sorted(frames.items())},
                high_precision_ngram_candidates=cues,
                high_precision_ngram_screen_is_diagnostic_not_proof=True)


def normalise(text: str) -> str:
    return ' '.join(re.findall(r"[a-z0-9']+", text.casefold()))


def materialise(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, trajectories = [], []
    for case in cases:
        topic = case['topic']
        scenario = 's15v4-'+stable(topic.key+':'+str(case['case'])+':'+case['act'])[:20]
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
            row = dict(id='s15v4-'+identity[:20], schema_version=VERSION, input=state,
                labels=labels, advice_present=labels['D2'],
                annotation=dict(profile=point['profile'], visible_evidence=point['evidence'],
                    rationale=RULES[point['profile']], status='construction_label_pending_independent_review'),
                metadata=dict(contrast_family=topic.key, split_group=topic.key, split=case['split'],
                    customer_disclosure=case['disclosure'], scenario_product_domains=[topic.domain],
                    coverage_boundary=case['act'], source_refs=[dict(scenario_id=scenario,
                        checkpoint_index=index, is_final=index==len(case['points'])-1, total_checkpoints=len(case['points']),
                        value_added=f'{case["act"]} / {topic.key} / customer situation {case["case"]}')],
                    recommendation_withdrawn=point['flags'].get('recommendation_withdrawn',False),
                    synthetic=True, author='codex_local_compositional_bootstrap',
                    construction_recipe=case['act'], surface_bank=case['act']+':'+str(point['frame_id']),
                    frame_id=point['frame_id'], customer_basis_type=point['basis_type'],
                    customer_basis_text=basis(topic, case['case'])[0],
                    recommendation_withdrawn_in_target=point['flags'].get('recommendation_withdrawn',False),
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
    if len(rows) != 15000:
        raise ValueError('Exactly 15,000 snapshots required')
    report = audit(rows, trajectories)
    report.update(acceptance(rows, trajectories))
    report.update(generation)
    report.update(screen(rows))
    report.update(generator=GENERATOR, authored_contrast_families=len(TOPICS),
        composition_trajectories=len(trajectories), independently_authored_trajectories=0,
        authored_speech_act_recipes=len(ACTS), split_specific_surface_banks=sum(len(p) for p in FRAME_POOLS.values())*len(ACTS),
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
        'Local compositional construction labels, not independently authored or reviewed scenarios.',
        'Sixty semantic topic families and thirty-four speech-act recipes are reused; 15,000 unique inputs do not imply 15,000 independent reasoning problems.',
        'Semantic topic families are partition-owned. All thirty-four speech-act recipes and some scaffolding remain shared; this is not an independent challenge benchmark.',
        'Lexical duplicate screening does not prove semantic independence.',
        'All recommendations are detection examples; their suitability, truth, authorisation and legal compliance are not adjudicated.',
        'No audio timing, intervention policy, native token-length validation, teacher verification or D1–D7 training run.']
    schema = yaml.safe_load((ROOT/'config/dataset/question_schema_streaming_v4.yaml').read_text())
    native_questions = {key: dict(type='noul',instructions=q['instructions'])
                        for key,q in schema['decisions'].items() if key != 'D7'}
    native_questions['D7'] = dict(type='choice',instructions=schema['decisions']['D7']['instructions'],criteria={
        'factual_information': 'Substantive facts or procedural help, with no product recommendation.',
        'general_advice': 'Product advice without individual suitability or demonstrated use of customer circumstances.',
        'personal_advice': 'Product advice asserting individual suitability or demonstrably using customer circumstances.',
        'other': 'No substantive facts, procedural help or product advice expressed in the available target.'})
    sources = [Path(__file__), Path(__file__).with_name('streaming_15k_catalog.py'),
        Path(__file__).with_name('streaming_v2.py'), Path(__file__).with_name('streaming_15k_v4_content.py'), Path(__file__).with_name('streaming_15k_v2.py'), Path(__file__).with_name('streaming_15k_long.py'), ROOT/'config/dataset/question_schema_streaming_v4.yaml',
        ROOT/'docs/annotation_standard_streaming_v4.md', ROOT/'scripts/templates/streaming_review_v2.html',
        ROOT/'scripts/build_streaming_15000_v4.py', ROOT/'scripts/validate_streaming_15000_v4.py',
        ROOT/'docs/streaming_15000_v4_release.md']
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
                records_jsonl_line=master_lines[r['id']],
                recommendation_withdrawn_in_target=r['metadata']['recommendation_withdrawn_in_target']) for i,r in enumerate(selected,1)])
        write_json(stage/'trajectories.json',trajectories)
        write_json(stage/'quality_report.json',report)
        write_json(stage/'questions.json',schema)
        write_json(stage/'native_questions.json',native_questions)
        write_json(stage/'line_index.json',{r['id']:i for i,r in enumerate(rows,1)})
        write_jsonl(stage/'review_queue.jsonl',[dict(record_id=r['id'],records_jsonl_line=i,
            review_status='pending_independent_annotation', review_reasons=[r['metadata']['coverage_boundary']],
            priority='high' if r['labels']['D6'] or r['metadata']['recommendation_withdrawn'] else 'normal') for i,r in enumerate(rows,1)])
        (stage/'annotation_standard.md').write_text((ROOT/'docs/annotation_standard_streaming_v4.md').read_text())
        payload=dict(rows=rows,questions=schema,report=report,
            dataset_id=stable(json.dumps(rows,sort_keys=True,ensure_ascii=False)))
        page=(ROOT/'scripts/templates/streaming_review_v2.html').read_text().replace('__DATASET_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
        page=page.replace('Version 2 ·','15,000-snapshot construction corpus · Version 4 ·')
        page=page.replace('authored trajectories', 'constructed trajectories').replace('End of authored turn', 'End of constructed turn')
        page=page.replace('href="/v1"', 'href="/v3"').replace('Previous review batch and its saved comments', 'Previous 15,000-snapshot v3 and its saved comments')
        page=page.replace('limitations:data.report.limitations',
            'checkpoint_counts:data.report.checkpoint_count_distribution,'
            'late_advice_fraction:data.report.final_checkpoint_first_advice_fraction,'
            'longer_trajectories:data.report.trajectories_over_three_checkpoints,'
            'long_trajectory_final_classes:data.report.long_trajectory_final_classes_by_split_and_length,'
            'preamble_outcomes:data.report.bridge_final_outcome_counts,'
            'D3_snapshots:data.report.D3_positive_snapshots_by_split,'
            'D3_distinct_contexts:data.report.D3_distinct_customer_contexts_by_split,'
            'D4_only_personal_fraction:data.report.personal_advice_D4_only_fraction,'
            'frame_coverage:data.report.observed_frame_ids,'
            'high_precision_phrases:data.report.high_precision_ngram_candidates,'
            'limitations:data.report.limitations')
        (stage/'review.html').write_text(page)
        baseline=ROOT/'data/generated/streaming_v2_review_batch_03/records.jsonl'
        (stage/'regression_reference').mkdir()
        (stage/'regression_reference/records.jsonl').write_bytes(baseline.read_bytes())
        (stage/'regression_reference/README.md').write_text(
            'The previously reviewed 518 snapshots are preserved here as a regression reference.\n'
            'They are excluded from the 15,000 records and from all new split/Clef exports.\n'
            'They are not an independently authored challenge set or legal approval.\n')
        (stage/'README.md').write_text((ROOT/'docs/streaming_15000_v4_release.md').read_text())
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
    actual_audit.update(acceptance(rows, trajectories))
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
        if len(selected) != n or len(exports) != len(selected) or len(index) != len(selected):
            raise ValueError('Native export count differs from accepted unique rows')
        for i,(row,export,ref) in enumerate(zip(selected,exports,index,strict=True),1):
            expected = dict(state=row['input']['customer_context']+'\nAGENT: '+row['input']['agent_response'],questions=native,gold=row['labels'])
            if export != expected or ref['record_id'] != row['id'] or ref['export_line'] != i:
                raise ValueError('Native export or lineage differs')
            if ref['recommendation_withdrawn_in_target'] != row['metadata']['recommendation_withdrawn_in_target']:
                raise ValueError('Withdrawal lineage flag differs')
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
