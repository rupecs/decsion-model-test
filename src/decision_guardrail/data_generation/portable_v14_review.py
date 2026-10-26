"""Review and blank-label annotation artifacts for an actual v14 candidate."""
from collections import Counter
import json

from .context_v5_review import digest, write_json, write_rows
from .context_v8_release import response_cues
from .context_v10_audits import low_support_cues
from ..training_readiness import annotation_fingerprint


def write_review_artifacts(root, stage, rows, designs, report):
    write_json(stage / "response_cue_audit.json", response_cues(rows))
    write_json(stage / "response_cues_low_support.json", low_support_cues(rows))
    write_rows(stage / "review_queue.jsonl", [dict(record_id=r["id"], input_labels_sha256=annotation_fingerprint(r),
        status="pending_independent_annotation", conversation_coherence=None, basis_relevance=None,
        adjudication_required=r["metadata"]["boundary_adjudication_required"]) for r in rows])
    write_rows(stage / "D3_only_group_adjudication.jsonl", [dict(record_id=r["id"],input=r["input"],
        proposed_labels=r["labels"],status="pending_independent_block_adjudication") for r in rows if r["labels"]["D3"] and not r["labels"]["D4"]])
    # Do not expose the construction order (fact/opinion/selection/prefix) to
    # annotators. IDs encode inputs, not proposed labels; order is deterministic.
    blinded=sorted(rows,key=lambda r:digest("v14-blank-order:"+r["id"]))
    write_rows(stage / "blank_annotation_packet.jsonl", [dict(record_id=r["id"],input=r["input"],
        assessed_labels={d:None for d in ("D1","D2","D3","D4","D5","D6","D7")},conversation_coherence=None,
        basis_relevance=None,evidence=[],reviewer=None,status="unreviewed",judge_available_text_only=True) for r in blinded])
    (stage / "README.md").write_text(f"# V14 construction candidates\n\n{len(rows)} snapshots from {len(designs)} authored design proposals. "
        "Zero independently approved training rows. Read quality_report.json, the cue diagnostics and annotation_standard.md. "
        "Native token coverage, approved D1-D7 GPU smoke/reload and independent challenge validation remain pending.\n")
    protocol=root / "docs/annotation_standard_streaming_v14.md"
    (stage / "annotation_standard.md").write_text(protocol.read_text())
    contexts=list(dict.fromkeys(r["input"]["customer_context"] for r in rows));lookup={text:i for i,text in enumerate(contexts)}
    view=[dict(id=r["id"],input=dict(customer_context=lookup[r["input"]["customer_context"]],agent_response=r["input"]["agent_response"]),
        labels=r["labels"],advice_present=r["advice_present"],annotation=dict(rationale=r["annotation"]["rationale"],visible_evidence=r["input"]["agent_response"]),
        metadata={k:r["metadata"][k] for k in ("contrast_family","split","recommendation_withdrawn","coverage_boundary","source_refs")}) for r in rows]
    summary={**report,**report.get("structural_report",{}),**report.get("naturalness",{}),
             "authored_contrast_families":len(designs),"credit_advice_rows":sum(r["labels"]["D6"] for r in rows)}
    payload=dict(rows=view,contexts=contexts,questions=json.loads((stage / "questions.json").read_text()),report=summary,dataset_id=digest(rows))
    page=(root / "data/authored/streaming_context_v14/review_shell.html").read_text()
    marker='<script id="dataset" type="application/json">';before,sep,tail=page.partition(marker);_,end,after=tail.partition('</script>')
    if not sep or not end:raise ValueError("Missing frozen review template boundary")
    page=(before+marker+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script>'+after)
    page=page.replace('Context v12','Context v14').replace('href="/v1"','href="http://127.0.0.1:8782/"')
    (stage / "review.html").write_text(page)
