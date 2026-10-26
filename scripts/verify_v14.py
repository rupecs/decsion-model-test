"""CPU-only data integrity, all native joins, leakage and dialogue audit."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from decision_guardrail.config import load_config
from decision_guardrail.dataset import load_splits
from decision_guardrail.data_generation.dialogue_release_audit import audit
from decision_guardrail.data_generation.context_v6_release import native_questions,validate_rows,normal_input

def verify():
 directory=ROOT/'data/generated/streaming_context_v14_release'
 with (directory/'manifest.json').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='90baf0e0e75c4c3331fe89c9d7b3f5c7e3427d6155078e990501efe322f3c281'
 cfg=load_config(experiment='streaming_context_v14')
 splits=load_splits(cfg)
 rows=[json.loads(x) for x in (directory/'records.jsonl').read_text().splitlines()]
 designs=[json.loads(x) for x in (directory/'designs.jsonl').read_text().splitlines()]
 bindings=json.loads((directory/'dialogue_bindings.json').read_text())
 report=audit(rows,set(),bindings,dict(minimum_contexts_per_call=2,minimum_designs_per_call=2,minimum_rows_per_class=30,minimum_other_fraction=.1),10000)
 assert not report['failures'],report['failures']
 _,qs=native_questions(ROOT);structural,paths=validate_rows(rows,[],qs)
 for field in ['customer_context','agent_response']:
  seen={}
  for row in rows:
   key=' '.join(row['input'][field].lower().split());split=row['metadata']['split']
   if key in seen and seen[key]!=split:raise ValueError('Cross-split text leakage')
   seen[key]=split
 assert len(rows)==17467 and len(designs)==356
 assert len({r['id'] for r in rows})==17467
 return dict(records=len(rows),designs=len(designs),splits={k:len(v) for k,v in splits.items()},native_joins_and_checksums='passed',dialogue_and_leakage_audit='passed',independently_approved_records=0,historical_exposure='Preserved publication evidence, not reasserted as fresh review')

if __name__=='__main__':print(json.dumps(verify(),indent=2))
