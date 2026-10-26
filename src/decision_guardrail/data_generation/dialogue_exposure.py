"""Cumulative, content-bound exposure of reviewed construction designs.

Content fingerprints detect identity despite renamed IDs; they do not prove
that a paraphrased scenario is semantically fresh. That still needs review.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .dialogue_release_audit import normalized, read_jsonl

LEDGER = "data/reviews/dialogue_exposure_ledger_v1.json"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def design_fingerprint(design: dict) -> str:
    """An ID-independent fingerprint of the declared substantive design."""
    fields = (design.get("product_topic", ""), design.get("customer_goal", design.get("actual_need", "")),
              design.get("decision_basis", design.get("reason", "")))
    return hashlib.sha256(json.dumps([normalized(x) for x in fields], separators=(",", ":")).encode()).hexdigest()


def design_ids(row: dict) -> set[str]:
    meta = row.get("metadata", {})
    ids = {meta.get("contrast_family"), meta.get("scenario_family_id"), row.get("family")}
    for item in meta.get("lineage", []):
        ids.update((item.get("authored_case"), item.get("source_design_id")))
    ids.update(meta.get("source_design_ids", []))
    # Merged rows can contain both A/B design IDs even if contrast_family is A.
    for unit in meta.get("units", []):
        if "-case:" in unit:
            ids.add(unit.split("-case:", 1)[1])
    return {value for value in ids if isinstance(value, str) and value}


def row_input(row: dict) -> dict:
    return row.get("input", row)


def review_paths(root: Path) -> list[Path]:
    return sorted((root / "data/reviews").rglob("record_review.jsonl"))


def source_paths(root: Path) -> list[Path]:
    return sorted({p for name in ("records.jsonl", "master_dataset.jsonl", "master.jsonl")
                   for p in (root / "data/generated").rglob(name)} |
                  set((root / "data/authoring").glob("*.jsonl")))


def build_ledger(root: Path) -> dict:
    reviews = review_paths(root)
    by_review = {}
    all_ids = set()
    for path in reviews:
        ids = {r.get("id", r.get("record_id")) for r in read_jsonl(path)}
        if None in ids:
            raise ValueError(f"Missing review record ID: {path}")
        by_review[str(path.relative_to(root))] = ids
        all_ids.update(ids)
    if not all_ids:
        raise ValueError("No recorded review exposure found")
    sources, found, contexts = {}, set(), set()
    aliases, representatives = set(), set()
    # Union lineage contributions, including shared IDs across old releases.
    for path in source_paths(root):
        matched = False
        for row in read_jsonl(path):
            key = row.get("id", row.get("record_id"))
            if key not in all_ids:
                continue
            matched = True
            found.add(key)
            aliases.update(design_ids(row))
            family = row.get("metadata", {}).get("contrast_family")
            if family:
                representatives.add(family)
            context = row_input(row).get("customer_context")
            if context:
                contexts.add(fingerprint(context))
        if matched:
            sources[str(path.relative_to(root))] = sha(path)
    unknown = all_ids - found
    if unknown:
        raise ValueError(f"Unresolved reviewed source IDs: {len(unknown)}; examples {sorted(unknown)[:5]}")
    # Content from every version of a reviewed design is exposed, not only the
    # version whose wording happened to be shown to the reviewer.
    goals, designs = set(), set()
    for path in sorted((root / "data/generated").rglob("author_case_quality_proposals.jsonl")):
        matched = False
        for proposal in read_jsonl(path):
            if proposal["case_id"] not in aliases:
                continue
            matched = True
            goals.add(fingerprint(proposal["actual_need"]))
            designs.add(design_fingerprint(proposal))
        if matched:
            sources[str(path.relative_to(root))] = sha(path)
    # New releases carry an explicit design registry rather than the legacy
    # product-only proposal file. Preserve its content anchors after review too.
    for path in sorted((root / "data/generated").rglob("designs.jsonl")):
        matched = False
        for design in read_jsonl(path):
            if design.get("design_id") not in aliases:
                continue
            matched = True
            goals.add(fingerprint(design["customer_goal"]))
            designs.add(design_fingerprint(design))
        if matched:
            sources[str(path.relative_to(root))] = sha(path)
    # Service families have no product proposal entry. Preserve their exact
    # source requests as content anchors as well as their lineage IDs.
    for path in source_paths(root):
        matched = False
        for row in read_jsonl(path):
            if not design_ids(row).intersection(aliases):
                continue
            matched = True
            inp = row_input(row)
            context = inp.get("customer_context", "")
            if context:
                contexts.add(fingerprint(context))
                for span in row.get("annotation", {}).get("evidence", {}).get("context_support_spans", []):
                    if span.get("text"):
                        goals.add(fingerprint(span["text"]))
        if matched:
            sources[str(path.relative_to(root))] = sha(path)
    return dict(format="cumulative_dialogue_exposure_v1", review_files={str(p.relative_to(root)): sha(p) for p in reviews},
                source_files=dict(sorted(sources.items())), reviewed_record_ids=sorted(all_ids),
                reviewed_design_ids=sorted(aliases), representative_design_ids=sorted(representatives),
                goal_fingerprints=sorted(goals), design_fingerprints=sorted(designs),
                context_fingerprints=sorted(contexts), unresolved_review_ids=[],
                review_record_counts={path: len(ids) for path, ids in by_review.items()},
                limitations=["Exact normalized fingerprints are not a semantic paraphrase detector.",
                             "Exposure records model review, not human annotation or legal approval."])


def load_ledger(root: Path) -> dict:
    policy = json.loads((root / "config/dataset/dialogue_exposure_policy.json").read_text())
    path = root / policy["ledger"]
    if not path.resolve().is_relative_to(root.resolve()) or sha(path) != policy["sha256"]:
        raise ValueError("Cumulative exposure ledger hash does not match the repository policy")
    ledger = json.loads(path.read_text())
    if ledger.get("format") != "cumulative_dialogue_exposure_v1" or not ledger.get("reviewed_record_ids"):
        raise ValueError("Missing or invalid cumulative review ledger")
    actual_reviews = {str(p.relative_to(root)) for p in review_paths(root)}
    if set(ledger["review_files"]) != actual_reviews:
        raise ValueError("Cumulative ledger omits a review or lists a missing review; rebuild it")
    for name, expected in {**ledger["review_files"], **ledger["source_files"]}.items():
        if sha(root / name) != expected:
            raise ValueError("Cumulative exposure source hash mismatch: " + name)
    if ledger.get("unresolved_review_ids"):
        raise ValueError("Cumulative exposure has unresolved reviewed records")
    return {**ledger, "ledger_path": str(path.relative_to(root)), "ledger_sha256": sha(path)}


def exposure_hits(rows: list[dict], ledger: dict, designs: list[dict] | None = None) -> list[dict]:
    ids = set(ledger["reviewed_design_ids"])
    record_ids = set(ledger.get("reviewed_record_ids", []))
    contexts = set(ledger["context_fingerprints"])
    goals = set(ledger["goal_fingerprints"])
    fingerprints = set(ledger["design_fingerprints"])
    by_id = {d["design_id"]: d for d in designs or []}
    hits = []
    for row in rows:
        reasons = []
        if row["id"] in record_ids:
            reasons.append("reviewed_record_id")
        candidate_ids = design_ids(row)
        if candidate_ids.intersection(ids):
            reasons.append("reviewed_design_lineage")
        if fingerprint(row_input(row)["customer_context"]) in contexts:
            reasons.append("reviewed_context_content")
        for design_id in candidate_ids:
            design = by_id.get(design_id)
            if not design:
                continue
            if design_fingerprint(design) in fingerprints:
                reasons.append("reviewed_design_content")
            if fingerprint(design["customer_goal"]) in goals:
                reasons.append("reviewed_customer_goal")
            if set(design.get("source_design_ids", [])).intersection(ids):
                reasons.append("reviewed_declared_ancestry")
        if reasons:
            hits.append(dict(record_id=row["id"], reasons=sorted(set(reasons))))
    return hits
