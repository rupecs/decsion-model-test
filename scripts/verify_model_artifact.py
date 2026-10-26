"""CPU-only native artifact verification; never loads original base weights."""
import hashlib,json,struct,sys
from pathlib import Path

def verify(checkpoint:Path,reports:Path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from decision_guardrail.checkpoint import verify_checkpoint
    verify_checkpoint(checkpoint)
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    for relative,expected in manifest.items():
        path=checkpoint/relative
        if not path.is_file() or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=expected:
            raise ValueError('Checkpoint checksum mismatch: '+relative)
    with (checkpoint/'joint_head.safetensors').open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if length>16000000:raise ValueError('Invalid tensor header')
        head=json.loads(f.read(length))
    dtypes={k:v['dtype'] for k,v in head.items() if k!='__metadata__'}
    if not dtypes or set(dtypes.values())!={'F32'}:raise ValueError('Saved head is not exact FP32')
    reload=json.loads((reports/'reload-verification.json').read_text())
    if reload['gpu_reload']!='passed' or reload['checked_questions']!=[f'D{i}' for i in range(1,8)]:
        raise ValueError('All-seven reload missing')
    metadata=json.loads((checkpoint/'metadata.json').read_text())
    if metadata['gradient_checks']!={'head_gradient':True,'lora_gradient':True,'unexpected_gradient':False}:
        raise ValueError('Real gradient checks did not pass')
    if metadata['parameter_updates']!={'head':True,'lora':True}:raise ValueError('Weights did not update')
    if not metadata['validation_selection']['best_state_restored']:raise ValueError('Best validation state not restored')
    return {'manifest_files':len(manifest),'head_tensors':len(dtypes),'head_dtype':'F32','all_seven_reload':'passed',
            'gradient_and_update_checks':'passed','best_state_restored':True}

if __name__=='__main__':
    result=verify(Path(sys.argv[1]),Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
