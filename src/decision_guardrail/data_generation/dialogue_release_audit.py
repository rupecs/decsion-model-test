"""Additional construction gates following the v11 review.

This is deliberately separate from frozen release verifiers. A valid checksum
does not imply representative evaluation or coherent dialogue. No check here
approves annotations, native token coverage, GPU work, or training.
"""
from collections import Counter, defaultdict
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any

SPLITS = ("train", "validation", "calibration", "test")
DOMAINS = ("credit", "deposit", "insurance", "investment", "payment", "super", "service")
CALLS = tuple(f"C{i:02}" for i in range(1, 21))
CLASSES = ("factual_information", "general_advice", "personal_advice", "other")


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


def turns(context: str) -> list[tuple[str, str]]:
    """Parse role boundaries, allowing a turn to contain newline continuations."""
    result: list[tuple[str, str]] = []
    for line in context.splitlines():
        match = re.match(r"^(CUSTOMER|AGENT):\s*(.*)$", line)
        if match:
            result.append((match[1], match[2]))
        elif result:
            role, text = result[-1]
            result[-1] = (role, text + "\n" + line)
        elif line.strip():
            raise ValueError("Context must start with a CUSTOMER or AGENT role")
    return result


def repeated_agent_turns(rows: list[dict]) -> list[dict]:
    """Flag copying a substantive earlier turn, even inside a longer target.

    Flag all classes, not just personal advice. This is a construction screen,
    not a claim that repeating an answer is always wrong in real conversations.
    """
    hits = []
    for row in rows:
        target = normalized(row["input"]["agent_response"])
        for index, (role, text) in enumerate(turns(row["input"]["customer_context"])):
            previous = normalized(text)
            if role == "AGENT" and len(previous.split()) >= 4 and f" {previous} " in f" {target} ":
                hits.append(dict(record_id=row["id"], earlier_turn_index=index,
                                 earlier_turn=text, final_class=row["labels"]["D7"]))
                break
    return hits


def copy_diagnostics(rows: list[dict]) -> dict:
    """Exact customer echoes and lexical near-copies, reported separately.

    Similarity is a screening signal, not a semantic equivalence assertion.
    Negation/number changes are not automatically called a paraphrase.
    """
    near, customer = [], []
    cache = {}
    def similar(previous: str, target: str) -> dict | None:
        key = (previous, target)
        if key in cache:
            return cache[key]
        a, b = previous.split(), target.split()
        result = None
        if min(len(a), len(b)) >= 6 and previous not in target:
            critical = lambda words: {w for w in words if w in ("not", "never", "no", "cannot", "can't", "without") or any(c.isdigit() for c in w)}
            if critical(a) == critical(b):
                grams = lambda words: {tuple(words[i:i+5]) for i in range(len(words)-4)}
                ag, bg = grams(a), grams(b)
                overlap = len(ag & bg) / len(ag) if ag else 0
                ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
                scaffolding = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "would"}
                ac, bc = [w for w in a if w not in scaffolding], [w for w in b if w not in scaffolding]
                stripped = SequenceMatcher(None, ac, bc, autojunk=False).ratio() if min(len(ac), len(bc)) >= 5 else 0
                if overlap >= .7 or ratio >= .78 or stripped >= .86:
                    result = dict(fivegram_coverage=round(overlap, 4), token_sequence_ratio=round(ratio, 4),
                                  scaffold_stripped_ratio=round(stripped, 4))
        cache[key] = result
        return result
    for row in rows:
        target = normalized(row["input"]["agent_response"])
        for index, (role, text) in enumerate(turns(row["input"]["customer_context"])):
            previous = normalized(text)
            if role == "CUSTOMER" and len(previous.split()) >= 4 and f" {previous} " in f" {target} ":
                customer.append(dict(record_id=row["id"], earlier_turn_index=index, final_class=row["labels"]["D7"]))
                break
        for index, (role, text) in enumerate(turns(row["input"]["customer_context"])):
            match = similar(normalized(text), target) if role == "AGENT" else None
            if match:
                near.append(dict(record_id=row["id"], earlier_turn_index=index,
                                 final_class=row["labels"]["D7"], **match))
                break
    return dict(near_duplicate_agent_turns=near, customer_turn_echoes=customer,
                near_agent_by_class=dict(Counter(h["final_class"] for h in near)),
                customer_echo_by_class=dict(Counter(h["final_class"] for h in customer)))


def coverage(rows: list[dict], minimum_contexts_per_call: int = 2,
             minimum_rows_per_class: int = 30, minimum_other_fraction: float = .10,
             minimum_designs_per_call: int = 2) -> tuple[dict, list[dict]]:
    """Check each partition independently; aggregate coverage cannot satisfy it.

    Count topics and contexts as well as snapshots so prefixes cannot manufacture
    support. Context counts are only an upper bound on semantic diversity.
    """
    failures, report = [], {}
    for split in SPLITS:
        selected = [row for row in rows if row["metadata"].get("split") == split]
        domain_topics: dict[str, set] = defaultdict(set)
        call_contexts: dict[str, set] = defaultdict(set)
        call_topics: dict[str, set] = defaultdict(set)
        call_designs: dict[str, set] = defaultdict(set)
        for row in selected:
            meta = row["metadata"]
            for domain in meta["scenario_product_domains"]:
                domain_topics[domain].add(meta["parent_topic"])
            call = meta["call_category"]
            # Strip only greetings and controlled disfluencies at turn starts.
            # This prevents "Hi"/"Hello" augmentation satisfying the minimum.
            history = re.sub(r"(?im)^(CUSTOMER:\s*)(?:(?:hi|hello|uh|um)[,\s]+)+", r"\1",
                             row["input"]["customer_context"])
            call_contexts[call].add(normalized(history))
            call_topics[call].add(meta["parent_topic"])
            # A/B aliases merged into one row do not inflate support. Only the
            # declared originating design is counted, and v12 validates it by
            # content against its design registry before publication.
            call_designs[call].add(meta.get("design_sha256", meta["contrast_family"]))
        classes = Counter(row["labels"]["D7"] for row in selected)
        missing_domains = [domain for domain in DOMAINS if not domain_topics[domain]]
        missing_calls = [call for call in CALLS if not call_contexts[call]]
        thin_calls = [call for call in CALLS if len(call_contexts[call]) < minimum_contexts_per_call]
        thin_designs = [call for call in CALLS if len(call_designs[call]) < minimum_designs_per_call]
        thin_classes = [label for label in CLASSES if classes[label] < minimum_rows_per_class]
        other_fraction = classes["other"] / len(selected) if selected else 0
        checks = (("missing_domains", missing_domains), ("missing_calls", missing_calls),
                  ("insufficient_call_contexts", thin_calls), ("insufficient_call_designs", thin_designs),
                  ("insufficient_class_rows", thin_classes))
        for check, value in checks:
            if value:
                failures.append(dict(check=check, split=split, actual=value))
        if other_fraction < minimum_other_fraction:
            failures.append(dict(check="other_class_support", split=split, actual=other_fraction,
                                 minimum=minimum_other_fraction))
        report[split] = dict(records=len(selected), product_topics=len({r["metadata"]["parent_topic"]
                              for r in selected if r["metadata"]["scenario_product_domains"] != ["service"]}),
                             domain_topic_counts={d: len(domain_topics[d]) for d in DOMAINS},
                             call_categories=sum(bool(call_contexts[c]) for c in CALLS),
                             call_context_counts={c: len(call_contexts[c]) for c in CALLS},
                             call_topic_counts={c: len(call_topics[c]) for c in CALLS},
                             call_design_counts={c: len(call_designs[c]) for c in CALLS},
                             class_counts=dict(classes), other_fraction=other_fraction)
    invalid_splits = sorted({str(r["metadata"].get("split")) for r in rows
                             if r["metadata"].get("split") not in SPLITS})
    if invalid_splits:
        failures.append(dict(check="invalid_partition", actual=invalid_splits))
    return report, failures


def reviewed_designs(source_rows: list[dict], review_records: list[dict]) -> set[str]:
    """Resolve exposure against the reviewed source, never a filtered candidate.

    Otherwise dropping just the reviewed row would hide exposure of its siblings.
    """
    exposed_ids = {row.get("id", row.get("record_id")) for row in review_records}
    unknown = exposed_ids - {row["id"] for row in source_rows}
    if unknown:
        raise ValueError("Review IDs are missing from the declared reviewed source")
    return {r["metadata"]["contrast_family"] for r in source_rows
            if r["id"] in exposed_ids or r["metadata"].get("source_case_design_reviewed")}


def dialogue_bindings(context: str, bindings: list[dict]) -> list[dict]:
    """Require explicit, exact Q/A bindings rather than sentence-position slots.

    Exact identity is mechanical. The author must still write a relevant answer;
    an independent annotator must judge semantic fit. We never infer that from
    lexical overlap or from an author's self-issued 'pass' flag.
    """
    history = turns(context)
    failures, seen = [], set()
    for binding in bindings:
        index = binding.get("answer_turn_index")
        if type(index) is not int or index <= 0 or index >= len(history) or index in seen:
            failures.append(dict(check="invalid_answer_turn_index", actual=index))
            continue
        seen.add(index)
        role, answer = history[index]
        previous_role, question = history[index - 1]
        if (role != "AGENT" or previous_role != "CUSTOMER" or
                question != binding.get("customer_turn") or answer != binding.get("agent_turn")):
            failures.append(dict(check="question_answer_binding_mismatch", answer_turn_index=index))
    missing = [i for i, (role, _) in enumerate(history) if role == "AGENT" and i not in seen]
    if missing:
        failures.append(dict(check="unbound_agent_turns", actual=missing))
    return failures


def audit(rows: list[dict], exposed_families: set[str] | None = None,
          binding_ledger: dict[str, list[dict]] | None = None,
          coverage_options: dict | None = None, minimum_records_exclusive: int = 10000) -> dict[str, Any]:
    partitions, failures = coverage(rows, **(coverage_options or {}))
    if len(rows) <= minimum_records_exclusive:
        failures.append(dict(check="minimum_records_exclusive", actual=len(rows), minimum=minimum_records_exclusive))
    repeats = repeated_agent_turns(rows)
    if repeats:
        failures.append(dict(check="copied_earlier_agent_turn", actual=len(repeats)))
    copies = copy_diagnostics(rows)
    for key in ("near_duplicate_agent_turns", "customer_turn_echoes"):
        if copies[key]:
            failures.append(dict(check=key, actual=len(copies[key])))
    exposed_families = exposed_families or set()
    retained = [r["id"] for r in rows if r["metadata"].get("source_case_design_reviewed") or
                r["metadata"]["contrast_family"] in exposed_families]
    if retained:
        failures.append(dict(check="reviewed_design_retained_in_candidate", actual=len(retained)))
    # The ledger is keyed to complete history hashes: a binding for a different
    # history or an earlier version cannot bless a rewritten conversation.
    binding_ledger = binding_ledger or {}
    histories = {r["input"]["customer_context"] for r in rows}
    binding_errors = []
    for context in sorted(histories):
        key = hashlib.sha256(context.encode()).hexdigest()
        for error in dialogue_bindings(context, binding_ledger.get(key, [])):
            binding_errors.append(dict(context_sha256=key, **error))
    if binding_errors:
        failures.append(dict(check="missing_or_invalid_dialogue_bindings", actual=len(binding_errors)))
    # Even correct bindings do not replace independent semantic annotation.
    return dict(records=len(rows), partitions=partitions, failures=failures,
                construction_gate_passed=not failures, training_ready=False,
                copied_earlier_agent_turns=repeats, retained_reviewed_design_ids=retained,
                exact_agent_by_class=dict(Counter(h["final_class"] for h in repeats)),
                copy_diagnostics=copies,
                dialogue_binding_errors=binding_errors,
                semantic_coherence="not_established_by_this_audit",
                required_before_training=["independent_labels_relevance_coherence_and_adjudication",
                                          "native_token_coverage", "approved_D1_D7_smoke_and_reload",
                                          "independent_challenge"],
                limitations=["Minimum topic/context counts do not establish semantic diversity.",
                             "Q/A bindings prove identity, not whether an answer is relevant.",
                             "An earlier-turn echo is a construction warning, not a legal label rule.",
                             "These checks supplement frozen integrity, leakage, prefix and shortcut checks."])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
