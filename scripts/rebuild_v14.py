"""Replay the frozen v14 release from its authoring inputs; no old datasets.

This reproduces an immutable release, not a fresh annotation/exposure approval.
Historical audit reports are preserved as historical evidence; checksums refuse
changed inputs or outputs. See docs/reproduction.md for the replay boundary.
"""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from decision_guardrail.data_generation.portable_v14 import assemble
from decision_guardrail.data_generation.context_v5_review import write_rows,write_json
from decision_guardrail.data_generation.context_v6_release import native_questions,validate_rows
from decision_guardrail.data_generation.portable_v14_review import write_review_artifacts
from decision_guardrail.dataset import state_text

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def rebuild(output):
 output=Path(output).resolve()
 if output.exists():raise ValueError('Choose a new output directory; frozen releases are never overwritten')
 for name,expected in json.loads((ROOT/'provenance/authoring_pins.json').read_text()).items():
  if sha(ROOT/name)!=expected:raise ValueError('Frozen authoring input changed: '+name)
 assets=json.loads((ROOT/'provenance/frozen_release_metadata.json').read_text());manifest=assets['manifest.json']
 rows,designs,bindings,assembly=assemble(ROOT)
 schema,qs=native_questions(ROOT)
 integrity,trajectories=validate_rows(rows,[],qs)
 if integrity!=assets['quality_report.json']['structural_report']:raise ValueError('Structural replay changed')
 output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='v14-replay-',dir=output.parent) as temp:
  stage=Path(temp)/'release';stage.mkdir()
  for name,value in [('records.jsonl',rows),('designs.jsonl',designs)]:write_rows(stage/name,value)
  for name,value in [('dialogue_bindings.json',bindings),('quality_report.json',assets['quality_report.json']),('trajectories.json',trajectories),('questions.json',schema),('native_questions.json',qs),('annotation_ledger.json',[]),('assembly_provenance.json',assembly),('comparison_correction_ledger.json',assets['comparison_correction_ledger.json'])]:write_json(stage/name,value)
  write_review_artifacts(ROOT,stage,rows,designs,assets['quality_report.json'])
  (stage/'README.md').write_text(assets['README.md'])
  index={r['id']:i for i,r in enumerate(rows,1)}
  for folder in ['splits','construction_clef','clef_index']:(stage/folder).mkdir()
  for split in ['train','validation','calibration','test']:
   selected=[r for r in rows if r['metadata']['split']==split]
   write_rows(stage/'splits'/f'{split}.jsonl',selected)
   write_rows(stage/'construction_clef'/f'{split}.jsonl',[dict(state=state_text(**r['input']),questions=qs,gold=r['labels']) for r in selected])
   write_rows(stage/'clef_index'/f'{split}.jsonl',[dict(export_line=i,record_id=r['id'],records_jsonl_line=index[r['id']],recommendation_withdrawn_in_target=r['metadata'].get('recommendation_withdrawn_in_target',False)) for i,r in enumerate(selected,1)])
  actual={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
  if actual!=set(manifest['files']):raise ValueError('Replay file inventory changed')
  for name,expected in manifest['files'].items():
   if sha(stage/name)!=expected:raise ValueError('Replay bytes differ: '+name)
  write_json(stage/'manifest.json',manifest)
  if sha(stage/'manifest.json')!='90baf0e0e75c4c3331fe89c9d7b3f5c7e3427d6155078e990501efe322f3c281':raise ValueError('Original manifest changed')
  stage.rename(output)
 return dict(records=len(rows),all_release_files_byte_identical=True,output=str(output),independently_approved_records=0)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();print(json.dumps(rebuild(args.output),indent=2))
