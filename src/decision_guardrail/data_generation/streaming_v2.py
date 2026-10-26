"""Build a small, auditable streaming schema/content review batch.

No GPU dependencies, teacher calls, stochastic annotation, or legacy-data edits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
import json
from pathlib import Path
import re
import tempfile

import yaml

from .streaming_v2_content import FAMILIES

ROOT = Path(__file__).resolve().parents[3]
VERSION = "banking_streaming_d1_d7_v2_binary_draft"
CLASSES = ("factual_information", "general_advice", "personal_advice", "other")
# Explicit annotation profiles, selected by the author for each checkpoint.
# D1–D6 are Boolean evidence-so-far decisions; D7 is a four-way class.
PROFILES = {
    "fragment": (False, False, False, False, False, False, "other"),
    "product_fragment": (True, False, False, False, False, False, "other"),
    "factual": (True, False, False, False, True, False, "factual_information"),
    "procedure": (False, False, False, False, True, False, "factual_information"),
    "other": (False, False, False, False, False, False, "other"),
    "product_other": (True, False, False, False, False, False, "other"),
    "general": (True, True, False, False, False, False, "general_advice"),
    "personal": (True, True, True, True, False, False, "personal_advice"),
    "suitability": (True, True, False, True, False, False, "personal_advice"),
    "credit_general": (True, True, False, False, False, True, "general_advice"),
    "credit_personal": (True, True, True, True, False, True, "personal_advice"),
    "credit_suitability": (True, True, False, True, False, True, "personal_advice"),
}
RULES = {
    "fragment": "No product advice or substantive help has been expressed in the available fragment. Do not infer its continuation.",
    "product_fragment": "A product is identifiable, but no advice or substantive help has yet been expressed.",
    "factual": "Substantive product facts or servicing help, without an endorsed product opinion.",
    "procedure": "Substantive process/security help, without a product recommendation or target product reference.",
    "other": "Resolved acknowledgement, clarification or other non-substantive content; no product advice.",
    "product_other": "A product reference with refusal, empathy or clarification; no endorsed product opinion.",
    "general": "A product opinion without an individual suitability verdict or use of customer circumstances.",
    "personal": "Product advice explicitly linked to the customer's disclosed circumstances.",
    "suitability": "An individual suitability verdict without observable reasoning from customer circumstances.",
    "credit_general": "A credit-product opinion without individual tailoring or a suitability verdict.",
    "credit_personal": "A credit recommendation explicitly uses the customer's circumstances.",
    "credit_suitability": "A credit suitability verdict for this customer without observable tailoring reasoning.",
}


def encode(profile: str) -> dict:
    return dict(zip((f"D{i}" for i in range(1, 8)), PROFILES[profile], strict=True))


def validate_labels(labels: dict) -> None:
    if set(labels) != {f"D{i}" for i in range(1, 8)}:
        raise ValueError("Exactly D1–D7 are required")
    for key in (f"D{i}" for i in range(1, 7)):
        if type(labels[key]) is not bool:
            raise ValueError(f"{key} must be a Boolean")
    if labels["D7"] not in CLASSES:
        raise ValueError("Unknown final class")
    if labels["D2"]:
        if not labels["D1"] or labels["D5"]:
            raise ValueError("Advice requires product scope and excludes facts-only")
        expected = "personal_advice" if labels["D3"] or labels["D4"] else "general_advice"
    else:
        if any(labels[f"D{i}"] for i in (3, 4, 6)):
            raise ValueError("D3, D4, D6 cannot assert advice when D2=false")
        expected = "factual_information" if labels["D5"] else "other"
    if labels["D7"] != expected:
        raise ValueError("Final class conflicts with supporting decisions")


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


def shingles(text: str) -> set[tuple[str, ...]]:
    words = normalise(text).split()
    return {tuple(words[i:i+4]) for i in range(max(0, len(words)-3))}


def author_review(families: list[dict]) -> None:
    ids = set()
    for family in families:
        if family["id"] in ids:
            raise ValueError("Duplicate family id")
        ids.add(family["id"])
        if not family["boundary"] or not family["context"].startswith("CUSTOMER:"):
            raise ValueError("Family needs a boundary and role-marked context")
        for variant in family["variants"]:
            if not variant["purpose"] or len(variant["checkpoints"]) < 2:
                raise ValueError("Each trajectory needs a purpose and multiple snapshots")
            previous = ""
            advice_seen = False
            for point in variant["checkpoints"]:
                text = point["text"]
                if not text or not text.startswith(previous) or text == previous:
                    raise ValueError("Snapshots must strictly extend the accumulated text")
                if not point["evidence"] or point["evidence"] not in text:
                    raise ValueError("Evidence must be visible in the current snapshot")
                labels = encode(point["profile"])
                validate_labels(labels)
                if advice_seen and labels["D2"] is not True:
                    raise ValueError("Do not erase previously exposed advice")
                advice_seen |= labels["D2"] is True
                if point["flags"].get("recommendation_withdrawn") and not advice_seen:
                    raise ValueError("Withdrawal flag without prior advice")
                previous = text


def grouped_families(families: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """Keep exact inputs and lexical-similar full targets in one split.

    The lexical screen is transparent and reproducible, not an embedding screen.
    Generic short fragments across unrelated contexts are audited but do not link
    every unrelated family together.
    """
    parent = {f["id"]: f["id"] for f in families}

    def root(key: str) -> str:
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    finals = []
    for family in families:
        for variant in family["variants"]:
            for point in variant["checkpoints"]:
                key = (normalise(family["context"]), normalise(point["text"]))
                if key in seen:
                    union(seen[key], family["id"])
                seen[key] = family["id"]
            finals.append((family["id"], variant["checkpoints"][-1]["text"]))
    candidates = []
    for i, (a, ta) in enumerate(finals):
        sa = shingles(ta)
        if len(normalise(ta).split()) < 8:
            continue
        for b, tb in finals[i+1:]:
            if a == b or len(normalise(tb).split()) < 8:
                continue
            sb = shingles(tb)
            score = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
            if score >= .35:
                union(a, b)
                candidates.append(dict(family_a=a, family_b=b, jaccard_4gram=round(score, 4),
                                       response_a=ta, response_b=tb, disposition="same_split_group"))
    return {key: root(key) for key in parent}, candidates


def assign_splits(groups: dict[str, str]) -> dict[str, str]:
    keys = sorted(set(groups.values()), key=lambda k: sha256((VERSION + k).encode()).hexdigest())
    if len(keys) < 4:
        return {family: "train" for family in groups}
    # Review partitions, not performance benchmarks. Keep actual counts in report.
    heldout = max(1, len(keys)//10)
    plan = ["test"]*heldout + ["validation"]*heldout + ["calibration"]*heldout
    plan += ["train"]*(len(keys)-len(plan))
    owners = dict(zip(keys, plan, strict=True))
    return {family: owners[group] for family, group in groups.items()}


def make_rows(families: list[dict]) -> tuple[list[dict], dict]:
    author_review(families)
    groups, candidates = grouped_families(families)
    splits = assign_splits(groups)
    rows = {}
    trajectories = []
    for family in families:
        for v, variant in enumerate(family["variants"], 1):
            scenario_id = f"sv2-{family['id']}-{v:02d}"
            path = []
            for i, point in enumerate(variant["checkpoints"]):
                labels = encode(point["profile"])
                state = dict(customer_context=family["context"], agent_response=point["text"])
                identity = sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                ref = dict(scenario_id=scenario_id, checkpoint_index=i,
                           is_final=i == len(variant["checkpoints"])-1,
                           total_checkpoints=len(variant["checkpoints"]),
                           value_added=variant["purpose"])
                if identity in rows:
                    row = rows[identity]
                    if row["labels"] != labels:
                        raise ValueError("Same visible input has conflicting gold labels")
                    row["metadata"]["source_refs"].append(ref)
                else:
                    row = dict(id="sv2-"+identity[:20], schema_version=VERSION, input=state,
                        labels=labels, annotation=dict(profile=point["profile"],
                            visible_evidence=point["evidence"], rationale=RULES[point["profile"]],
                            status="construction_label_pending_independent_review"),
                        metadata=dict(contrast_family=family["id"], split_group=groups[family["id"]],
                            split=splits[family["id"]], customer_disclosure=family["disclosure"],
                            scenario_product_domains=family["products"],
                            coverage_boundary=family["boundary"], source_refs=[ref],
                            recommendation_withdrawn=point["flags"].get("recommendation_withdrawn", False),
                            synthetic=True, author="codex_local_authored_bootstrap",
                            regulatory_scope_is_separate_from_policy=True))
                    rows[identity] = row
                path.append(row["id"])
            trajectories.append(dict(id=scenario_id, family=family["id"],
                                     split=splits[family["id"]], snapshot_ids=path))
    rows = list(rows.values())
    for row in rows:
        row["advice_present"] = row["labels"]["D2"]
    report = audit(rows, trajectories)
    report["lexical_near_duplicate_candidates"] = candidates
    report["source_snapshots_before_dedup"] = sum(len(v["checkpoints"]) for f in families for v in f["variants"])
    report["deduplicated_snapshots"] = report["source_snapshots_before_dedup"] - len(rows)
    report["authored_contrast_families"] = len(families)
    report["split_groups"] = len(set(groups.values()))
    report["lexical_baselines"] = {
        field: lexical_baseline(rows, field) for field in ("customer_context", "agent_response")
    }
    report["trajectories"] = trajectories
    return rows, report


def lexical_baseline(rows: list[dict], field: str) -> dict:
    """Small diagnostic multinomial NB; results are not a robustness certificate."""
    train = [r for r in rows if r["metadata"]["split"] == "train"]
    documents = Counter(r["labels"]["D7"] for r in train)
    words = {c: Counter() for c in CLASSES}
    for row in train:
        words[row["labels"]["D7"]].update(normalise(row["input"][field]).split())
    vocab = set().union(*(set(w) for w in words.values()))
    sizes = {c: sum(w.values()) + max(1, len(vocab)) for c, w in words.items()}
    results = {}
    for split in ("validation", "calibration", "test"):
        test = [r for r in rows if r["metadata"]["split"] == split]
        confusion = Counter()
        for row in test:
            tokens = normalise(row["input"][field]).split()
            scores = {c: math.log((documents[c]+1)/(len(train)+len(CLASSES))) +
                      sum(math.log((words[c][w]+1)/sizes[c]) for w in tokens if w in vocab)
                      for c in CLASSES}
            predicted = max(CLASSES, key=lambda c: scores[c])
            confusion[row["labels"]["D7"], predicted] += 1
        correct = sum(n for (gold, predicted), n in confusion.items() if gold == predicted)
        counts = Counter(r["labels"]["D7"] for r in test)
        results[split] = dict(n=len(test), correct=correct,
            accuracy=correct/len(test) if test else None,
            heldout_majority_accuracy=max(counts.values())/len(test) if test else None,
            confusion={gold+" -> "+pred: n for (gold, pred), n in sorted(confusion.items())})
    return dict(method="Laplace-smoothed unigram multinomial naive Bayes, train partition only",
                diagnostic_only=True, results=results)


def audit(rows: list[dict], trajectories: list[dict]) -> dict:
    ids = {}
    groups = defaultdict(set)
    contexts = defaultdict(set)
    targets = defaultdict(list)
    for line, row in enumerate(rows, 1):
        validate_labels(row["labels"])
        if type(row.get("advice_present")) is not bool or row["advice_present"] != row["labels"]["D2"]:
            raise ValueError("Binary advice result disagrees with D2")
        if row["id"] in ids:
            raise ValueError("Duplicate row id")
        ids[row["id"]] = row
        groups[row["metadata"]["split_group"]].add(row["metadata"]["split"])
        contexts[row["input"]["customer_context"]].add(row["labels"]["D7"])
        targets[normalise(row["input"]["agent_response"])].append((line, row))
        if row["annotation"]["visible_evidence"] not in row["input"]["agent_response"]:
            raise ValueError("Annotation evidence is not in input")
        if set(row["input"]) != {"customer_context", "agent_response"}:
            raise ValueError("Model input contains unexpected fields or future text")
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("Group leakage")
    transitions = Counter()
    for trajectory in trajectories:
        path = [ids[key] for key in trajectory["snapshot_ids"]]
        for a, b in zip(path, path[1:]):
            if not b["input"]["agent_response"].startswith(a["input"]["agent_response"]):
                raise ValueError("Non-prefix trajectory")
            transitions[a["labels"]["D7"]+" -> "+b["labels"]["D7"]] += 1
        if any(row["metadata"]["split"] != trajectory["split"] for row in path):
            raise ValueError("Trajectory split leakage")
    split_counts = {split: dict(Counter(r["labels"]["D7"] for r in rows if r["metadata"]["split"] == split))
                    for split in ("train", "validation", "calibration", "test")}
    duplicate_targets = [dict(lines=[line for line, _ in matches],
        classes=sorted({r["labels"]["D7"] for _, r in matches}),
        splits=sorted({r["metadata"]["split"] for _, r in matches}),
        text=matches[0][1]["input"]["agent_response"])
        for matches in targets.values() if len(matches) > 1]
    # Transparent candidate cues, never gold-label rules. Precision alone is not a
    # pass/fail guarantee: some phrases carry real substantive meaning.
    cue_report = {}
    for phrase in ("given your", "for you", "i recommend", "this is general information",
                   "credit card", "generally", "could you", "[unintelligible]"):
        matches = [r for r in rows if phrase in r["input"]["agent_response"].casefold()]
        counts = Counter(r["labels"]["D7"] for r in matches)
        cue_report[phrase] = dict(n=len(matches), by_class=dict(counts),
            majority_precision=max(counts.values())/len(matches) if matches else None)
    length_bands = {}
    for c in CLASSES:
        lengths = [len(r["input"]["agent_response"].split()) for r in rows if r["labels"]["D7"] == c]
        length_bands[c] = dict(short_1_15=sum(n <= 15 for n in lengths),
            medium_16_79=sum(16 <= n < 80 for n in lengths), long_80_plus=sum(n >= 80 for n in lengths),
            max_words=max(lengths, default=0))
    return dict(schema_version=VERSION, records=len(rows), scenarios=len(trajectories),
        class_counts=dict(Counter(r["labels"]["D7"] for r in rows)), split_class_counts=split_counts,
        atomic_counts={f"D{i}": dict(Counter(str(r["labels"][f"D{i}"]).lower() for r in rows)) for i in range(1, 7)},
        partial_advice=sum(r["advice_present"] and any(not x["is_final"] for x in r["metadata"]["source_refs"]) for r in rows),
        partial_no_advice=sum(not r["advice_present"] and any(not x["is_final"] for x in r["metadata"]["source_refs"]) for r in rows),
        binary_counts=dict(Counter("advice" if r["advice_present"] else "no_advice" for r in rows)),
        context_class_counts=dict(Counter(str(len(c)) for c in contexts.values())),
        contexts_with_multiple_resolved_classes=sum(len(c) > 1 for c in contexts.values()),
        disclosure_by_class={c: dict(Counter(str(r["metadata"]["customer_disclosure"]).lower() for r in rows if r["labels"]["D7"] == c)) for c in CLASSES},
        credit_advice_rows=sum(r["labels"]["D6"] is True for r in rows),
        withdrawn_recommendation_rows=sum(r["metadata"]["recommendation_withdrawn"] for r in rows),
        transition_counts=dict(transitions), repeated_normalised_targets=duplicate_targets,
        phrase_screen=cue_report, structural_checks_passed=True,
        response_word_length_by_class=length_bands,
        missing_classes_by_partition={s: [c for c in CLASSES if not counts.get(c)] for s, counts in split_counts.items()},
        review_readiness="draft_only_not_a_training_release", independent_annotation=False,
        semantic_embedding_screen=False,
        limitations=["All labels are author construction labels; no independent or legal review.",
            "Small authored review partitions are not an independent performance benchmark.",
            "Lexical full-response screen groups candidates; semantic paraphrases can remain.",
            "Repeated short targets across different contexts are reported, not treated as independent novelty.",
            "Coverage remains uneven; no 15,000-row quota has been filled with paraphrases.",
            "No audio timing, intervention labels, native tokenisation, GPU test or D1–D7 training integration."])


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def build(output: Path, families: list[dict] | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new destination; existing outputs are never overwritten")
    families = FAMILIES if families is None else families
    rows, report = make_rows(families)
    sources = [Path(__file__), Path(__file__).with_name("streaming_v2_content.py"),
               ROOT/"docs/annotation_standard_streaming_v2.md",
               ROOT/"config/dataset/question_schema_streaming_v2.yaml",
               ROOT/"scripts/build_streaming_v2.py",
               ROOT/"scripts/templates/streaming_review_v2.html",
               Path(__file__).with_name("streaming_v1_content.py")]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="streaming-review-", dir=output.parent) as tmp:
        stage = Path(tmp)/"result"
        stage.mkdir()
        write_jsonl(stage/"records.jsonl", rows)
        for split in report["split_class_counts"]:
            write_jsonl(stage/"splits"/(split+".jsonl"), [r for r in rows if r["metadata"]["split"] == split])
        write_json(stage/"trajectories.json", report.pop("trajectories"))
        write_json(stage/"quality_report.json", report)
        schema = yaml.safe_load(sources[3].read_text())
        write_json(stage/"questions.json", schema)
        (stage/"annotation_standard.md").write_text(sources[2].read_text())
        # JSON in a non-executable script tag; escape HTML script terminators.
        dataset_id = sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        payload = json.dumps(dict(rows=rows, questions=schema, report=report, dataset_id=dataset_id), ensure_ascii=False).replace("<", "\\u003c")
        template = sources[5].read_text()
        (stage/"review.html").write_text(template.replace("__DATASET_JSON__", payload))
        write_json(stage/"line_index.json", {r["id"]: i for i, r in enumerate(rows, 1)})
        baseline = ROOT/"data/generated/streaming_v1_review_batch_02/records.jsonl"
        previous = {r['id'].replace('sv1-', 'sv2-', 1): (i, r)
                    for i, line in enumerate(baseline.read_text().splitlines(), 1)
                    for r in [json.loads(line)]} if baseline.exists() else {}
        changes = []
        for line, row in enumerate(rows, 1):
            if row['id'] not in previous:
                continue
            old_line, old = previous[row['id']]
            diff = {key: dict(before=old['labels'][key], after=value)
                    for key, value in row['labels'].items() if old['labels'][key] != value}
            if diff:
                changes.append(dict(old_record_id=old['id'], old_line=old_line,
                    new_record_id=row['id'], new_line=line, changes=diff,
                    reason='User-directed Boolean evidence-so-far policy; fragment D1 adjudicated separately.'))
        write_jsonl(stage/'changes_from_v1.jsonl', changes)
        write_jsonl(stage/"review_queue.jsonl", [dict(record_id=r["id"], records_jsonl_line=i,
            review_status="pending_independent_annotation", review_reasons=[r["metadata"]["coverage_boundary"]],
            priority="high" if r["labels"]["D6"] is True or r["metadata"]["recommendation_withdrawn"] else "normal")
            for i, r in enumerate(rows, 1)])
        manifest = dict(schema_version=VERSION, records=len(rows), scenarios=report["scenarios"],
            status=report["review_readiness"], source_hashes={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sources},
            files={str(p.relative_to(stage)): sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob("*")) if p.is_file()})
        manifest['baseline_records_sha256'] = sha256(baseline.read_bytes()).hexdigest() if baseline.exists() else None
        write_json(stage/"manifest.json", manifest)
        stage.rename(output)
    return report


def validate_bundle(output: Path) -> dict:
    """Validate frozen exports independently of building from current seeds."""
    output = Path(output).resolve()
    manifest = json.loads((output/"manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError("Manifest path outside bundle or missing file")
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    rows = [json.loads(line) for line in (output/"records.jsonl").read_text().splitlines()]
    paths = json.loads((output/"trajectories.json").read_text())
    report = audit(rows, paths)
    if manifest["records"] != len(rows) or manifest["scenarios"] != len(paths):
        raise ValueError("Manifest record/trajectory counts differ")
    index = json.loads((output/"line_index.json").read_text())
    if index != {row["id"]: line for line, row in enumerate(rows, 1)}:
        raise ValueError("Line index differs from records.jsonl")
    for split in ("train", "validation", "calibration", "test"):
        actual = [json.loads(line) for line in (output/"splits"/(split+".jsonl")).read_text().splitlines()]
        if actual != [r for r in rows if r["metadata"]["split"] == split]:
            raise ValueError("Split export differs from unique accepted rows")
    saved = json.loads((output/"quality_report.json").read_text())
    for key, value in report.items():
        if saved[key] != value:
            raise ValueError(f"Audit report mismatch: {key}")
    return dict(status="passed", records=len(rows), trajectories=len(paths),
                hashed_artifacts=len(manifest["files"]), independent_or_legal_review=False)
