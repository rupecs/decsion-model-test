"""Adapter-only checkpoints, integrity manifests and validated metadata relocation."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
from .storage import base_identity, file_hash
from .utils import read_json, write_json, fingerprint
from .schema import D1_D6, D1_D7, STREAMING_SCHEMAS, contract, contract_hash

REQUIRED = ['adapter_config.json','adapter_model.safetensors','joint_head.safetensors','joint_head_config.json','unsloth_decision_config.json','tokenizer_config.json','calibration.json','resolved_config.json','metadata.json','base_identity.json']


def verify_checkpoint(checkpoint: str | Path, config: dict | None = None) -> dict:
    """Verify completion and every saved file; optionally validate the active schema."""
    root=Path(checkpoint)
    for name in REQUIRED+['manifest.json']:
        if not (root/name).is_file():
            raise FileNotFoundError('Incomplete checkpoint: '+name)
    manifest=read_json(root/'manifest.json')
    if not set(REQUIRED).issubset(manifest):
        raise ValueError('Checkpoint manifest omits required assets')
    for name,digest in manifest.items():
        from .storage import safe_member
        if file_hash(safe_member(root,name))!=digest:
            raise ValueError('Checkpoint checksum mismatch: '+name)
    saved=read_json(root/'resolved_config.json')
    if config and contract(config) != contract(saved):
        raise ValueError('Active classification schema differs from checkpoint')
    if saved['dataset'].get('schema') in (D1_D6, *STREAMING_SCHEMAS):
        for name in ('decision_schema.json', 'split_assignments.json'):
            if name not in manifest or not (root/name).is_file():
                raise ValueError('D1–D6 checkpoint manifest omits schema/provenance assets')
        if read_json(root/'decision_schema.json') != contract(saved):
            raise ValueError('Saved decision schema differs from resolved configuration')
        if read_json(root/'calibration.json').get('schema_sha256') != contract_hash(saved):
            raise ValueError('Calibration decision schema differs from checkpoint')
    decision=read_json(root/'unsloth_decision_config.json')
    state=read_json(root/'calibration.json')['state']
    for key,value in state.items():
        if decision.get(key)!=value:
            raise ValueError('Saved calibration state was not preserved: '+key)
    return saved


@contextmanager
def relocated_checkpoint(checkpoint: str | Path, base: Path):
    """Relocate both base references in a disposable copy; source stays immutable.

    Weight files are symlinked read-only by convention to avoid copying adapters.
    Only metadata copies are edited. The context remains alive until loading ends.
    """
    root=Path(checkpoint).resolve(); verify_checkpoint(root)
    if read_json(root/'base_identity.json')!=base_identity(base):
        raise ValueError('Active base weights differ from the training base')
    with tempfile.TemporaryDirectory(prefix='.relocated-',dir=root.parent) as folder:
        target=Path(folder)
        for source in root.iterdir():
            if source.is_file():
                if source.suffix=='.json':
                    shutil.copy2(source,target/source.name)
                else:
                    (target/source.name).symlink_to(source)
        for name,key in [('adapter_config.json','base_model_name_or_path'),('unsloth_decision_config.json','base_model')]:
            value=read_json(target/name); value[key]=str(base.resolve()); write_json(target/name,value)
        # This copy intentionally has changed metadata and is not a distribution artifact.
        (target/'manifest.json').unlink()
        yield target


def save_checkpoint(model, processor, config: dict, base: Path, calibration: dict, metadata: dict, split_assignments: dict, destination: Path | None = None) -> Path:
    """Use native adapter saving and publish a complete checkpoint atomically."""
    destination=destination or Path(config['paths']['output_dir'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Output exists: {destination}; choose a new paths.output_dir')
    if not hasattr(model.encoder,'peft_config'):
        raise RuntimeError('Adapters required to avoid saving the full pretrained base')
    with tempfile.TemporaryDirectory(prefix='.saving-',dir=destination.parent) as folder:
        stage=Path(folder)/'checkpoint'
        # Source inspection: native adapter save stores serving head weights in reduced
        # precision. Preserve exact FP32 head via the native exact-saving implementation.
        from unsloth.models.decision import _save_clef_adapter
        _save_clef_adapter(model,str(stage),processor,exact=True)
        for name,value in {'resolved_config.json':config,'decision_schema.json':contract(config),'metadata.json':metadata,'calibration.json':calibration,'base_identity.json':base_identity(base),'split_assignments.json':split_assignments}.items():
            write_json(stage/name,value)
        manifest={str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
        write_json(stage/'manifest.json',manifest)
        verify_checkpoint(stage,config)
        if destination.exists():
            destination.rmdir()
        stage.replace(destination)
    return destination


def smoke_key(config: dict, identity: dict) -> str:
    """Tie smoke acceptance to architecture, actual base, hardware and optimization."""
    return fingerprint({'model':{k:v for k,v in config['model'].items() if k not in {'local_path','cache_dir','s3_uri','source'}},'head':config['decision_head'],'lora':config['lora'],'compute':config['compute_profile'],'identity':identity,'schema':contract(config),'head_only':config['training']['head_only'],'batch':config['training']['batch_size'],'accumulation':config['training']['gradient_accumulation_steps'],'learning_rate':config['training']['learning_rate'],'head_learning_rate':config['training']['head_learning_rate']})


@contextmanager
def stale_path_checkpoint(checkpoint: Path):
    """Build a disposable portability-test artifact with nonexistent base references."""
    with tempfile.TemporaryDirectory(prefix='.portability-',dir=checkpoint.parent) as folder:
        target=Path(folder)
        for source in checkpoint.iterdir():
            if source.is_file():
                if source.suffix=='.json':
                    shutil.copy2(source,target/source.name)
                else:
                    # Hardlinks keep integrity validation self-contained without copying weights.
                    import os
                    try:
                        os.link(source.resolve(),target/source.name)
                    except OSError:
                        shutil.copy2(source,target/source.name)
        for name,key in [('adapter_config.json','base_model_name_or_path'),('unsloth_decision_config.json','base_model')]:
            value=read_json(target/name); value[key]='/nonexistent-original-training-machine/gemma-4-E4B-it'; write_json(target/name,value)
        manifest={p.name:file_hash(p) for p in target.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(target/'manifest.json',manifest)
        yield target
