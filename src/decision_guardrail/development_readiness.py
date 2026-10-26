"""One authorised, hash-pinned synthetic development run; no data approval."""
import hashlib,json,os
from pathlib import Path
from .config import ROOT
from .schema import contract_hash

# Populated from the frozen local artifacts by the preparation step.
PINS = json.loads((Path(__file__).resolve().parents[2]/'development_pins.json').read_text())
DEV_ID='v14_synthetic_development_20261026'
SMOKE_ID='v14_d1_d7_integration_20261026'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def denied(reason):return dict(ready=False,status=reason,approved_records=0)


def check(config,smoke=False):
    t=config['training'];d=config['dataset'];m=config['model']
    if os.environ.get('DG_V14_SYNTHETIC_DEVELOPMENT')!=DEV_ID:return denied('explicit_synthetic_development_opt_in_required')
    if sha(ROOT/'development_authorization.json')!=PINS['authorization']:return denied('development_authorization_changed')
    if t.get('development_run') is not None and t.get('fixture_smoke') is not None:return denied('ambiguous_exception')
    if (m['compile'] is not False or m['dtype']!='bfloat16' or m['max_seq_length']!=2048 or m['load_in_4bit'] or m['load_in_8bit']
        or config['decision_head']['dtype']!='float32' or config['compute_profile'] not in {'a40_48gb','l40s_48gb','a100_80gb'}
        or t['head_only'] is not False or type(t['batch_size']) is not int or t['batch_size']!=1
        or type(t['gradient_accumulation_steps']) is not int or t['gradient_accumulation_steps']!=8
        or t['learning_rate']!=0.0002 or t['head_learning_rate'] is not None
        or config['lora']!=PINS['lora'] or contract_hash(config)!=PINS['contract']
        or d.get('path') is not None or d.get('multi_question') is not False):return denied('development_execution_contract_mismatch')
    selection=t.get('validation_selection',{})
    if selection!={'enabled':True,'patience':3,'min_delta':0.001,'restore_best':True}:return denied('validation_selection_contract_mismatch')
    if smoke:
        if (t.get('fixture_smoke')!=SMOKE_ID or config['experiment']!='smoke_test' or type(t['max_steps']) is not int
            or t['max_steps']!=3 or type(t['epochs']) is not int or t['epochs']!=1
            or t['evaluation_interval']!=1):return denied('development_smoke_bounds_mismatch')
        prefix='data/samples/v14_development_smoke';pins=PINS['smoke'];counts=dict.fromkeys(('train','validation','calibration','test'),5)
    else:
        if (t.get('development_run')!=DEV_ID or config['experiment']!='streaming_context_v14_development'
            or type(t['max_steps']) is not int or t['max_steps']!=2972
            or type(t['epochs']) is not int or t['epochs']!=2 or t['evaluation_interval']!=100
            or t['require_smoke'] is not True):return denied('development_training_bounds_mismatch')
        prefix='data/generated/streaming_context_v14_release';pins=PINS['full'];counts={'train':11883,'validation':1640,'calibration':2071,'test':1873}
    if (d.get('schema')!='banking_streaming_d1_d7_v6_context_draft' or d.get('splits')!=counts
        or d.get('expected_total')!=sum(counts.values()) or set(d.get('files',{}))!=set(counts)):
        return denied('development_data_contract_mismatch')
    directory=ROOT/prefix
    if sha(directory/'manifest.json')!=pins['manifest']:return denied('development_manifest_changed')
    if (sha(directory/'records.jsonl')!=pins['records'] or d.get('manifest_sha256')!=pins['manifest']
                      or d.get('provenance_sha256')!=pins['records']):return denied('development_source_changed')
    if (Path(d['manifest_path']).resolve()!=(directory/'manifest.json').resolve() or Path(d['provenance_path']).resolve()!=(directory/'records.jsonl').resolve()):return denied('development_source_path_changed')
    if type(t['evaluation_interval']) is not int or t['logging_steps']!=(1 if smoke else 5):return denied('development_observation_contract_mismatch')
    for split in counts:
        expected=(directory/'construction_clef'/f'{split}.jsonl').resolve();actual=Path(d['files'][split]).resolve()
        if actual!=expected or sha(actual)!=pins['files'][split]:return denied('development_native_export_changed')
    return dict(ready=True,status='hash_pinned_development_smoke_only' if smoke else 'hash_pinned_synthetic_development_only',
        approved_records=0,production_ready=False,annotation_approved=False,max_optimizer_steps=3 if smoke else 2972,
        reason='Explicit operator opt-in for this frozen synthetic experiment only. No independent annotation approval.')
