"""Offline setup and actionable environment checks without loading weights."""
import importlib.metadata as metadata
import inspect
import os
from pathlib import Path
import platform
import sys
import tempfile


def set_offline() -> None:
    """Enforce offline repository access before any model library import."""
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','DO_NOT_TRACK','WANDB_DISABLED']:
        os.environ[key]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['UNSLOTH_DISABLE_STATISTICS']='1'
    os.environ['WANDB_MODE']='disabled'


def configure_runtime(config: dict) -> None:
    """Select Unsloth compilation before importing the GPU runtime."""
    set_offline()
    os.environ['UNSLOTH_COMPILE_DISABLE'] = '0' if config['model'].get('compile', True) else '1'


def inspect_api() -> dict:
    """Inspect installed, exact-version Unsloth and required call signatures."""
    set_offline()
    if metadata.version('unsloth') != '2026.10.6':
        raise RuntimeError('Install unsloth==2026.10.6; other versions are unvalidated')
    from unsloth import FastDecisionModel, DecisionTrainer
    from unsloth.models.decision_from_lm import load_lm_as_decision_model
    required = {'from_pretrained':['model_name','dtype','local_files_only'], 'get_peft_model':['model','r','target_modules'], 'build_dataset':['rows','tokenizer','model'], 'calibrate':['model','tokenizer','items'], 'evaluate':['model','tokenizer','items'], 'predict':['model','tokenizer','state','questions'], 'for_inference':['model']}
    result = {}
    for name, parameters in required.items():
        sig = inspect.signature(getattr(FastDecisionModel,name))
        if not set(parameters).issubset(sig.parameters):
            raise RuntimeError(f'Unsupported FastDecisionModel.{name}{sig}')
        result[name]=str(sig)
    sig = inspect.signature(load_lm_as_decision_model)
    if not {'decision_head','head_init'}.issubset(sig.parameters):
        raise RuntimeError('Installed Unsloth lacks new Clef head construction')
    from unsloth.models.decision import _save_clef_adapter
    if not {'self','save_directory','tokenizer','exact'}.issubset(inspect.signature(_save_clef_adapter).parameters):
        raise RuntimeError('Exact FP32 native adapter saving API unavailable')
    result['exact_saving']=str(inspect.signature(_save_clef_adapter))
    result['DecisionTrainer']=str(inspect.signature(DecisionTrainer))
    return result


def preflight(config: dict, check_model: bool = True) -> dict:
    """Collect failures independently; never load a complete model here."""
    configure_runtime(config)
    result = {'python':sys.version, 'host':platform.platform(), 'packages':{}, 'errors':[], 'warnings':[], 'platform':config['platform'],'compute_profile':config['compute_profile'], 'model_compile':config['model'].get('compile', True)}
    if not (3,11) <= sys.version_info[:2] <= (3,13):
        result['warnings'].append('GPU environment should use Python 3.11–3.13; local CPU tests may use newer Python')
    for name in ['torch','unsloth','transformers','peft','datasets','accelerate','unsloth_zoo']:
        try:
            result['packages'][name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            result['errors'].append(f'Missing {name}; install the GPU extra in an approved CUDA environment')
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
        for raw in metadata.requires('unsloth') or []:
            requirement=Requirement(raw)
            if requirement.name.lower().replace('_','-') not in {'torch','transformers','peft','unsloth-zoo','datasets'}:
                continue
            if requirement.marker is not None and not requirement.marker.evaluate({'extra':''}):
                continue
            actual=metadata.version(requirement.name)
            if Version(actual) not in requirement.specifier:
                result['errors'].append(f'Incompatible {requirement.name}=={actual}; upstream requires {requirement.specifier}')
    except metadata.PackageNotFoundError:
        pass
    try:
        # Unsloth must precede transformers/peft imports for its patches.
        result['api']=inspect_api()
    except Exception as exc:
        result['errors'].append(f'Unsloth compatibility: {exc}')
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use a provisioned NVIDIA GPU')
        if torch.cuda.device_count()!=1:
            raise RuntimeError('Single-GPU only: expose exactly one GPU with CUDA_VISIBLE_DEVICES')
        free,total = torch.cuda.mem_get_info()
        result['gpu']={'name':torch.cuda.get_device_name(0),'free_gb':free/2**30,'total_gb':total/2**30,'cuda':torch.version.cuda}
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Native BF16 unavailable; choose compatible hardware')
        expected = config['compute']['gpu_name']
        if expected not in result['gpu']['name']:
            result['errors'].append(f'Allocated GPU differs from selected profile {expected}; select the actual compute profile')
        if free/2**30 < config['compute']['minimum_free_gb']:
            result['warnings'].append('Low available VRAM; loading/training may fail; choose more memory or reduce sequence length')
    except Exception as exc:
        result['errors'].append(str(exc))
    if config['compute']['nominal_vram_gb'] <=24:
        result['warnings'].append('24 GB BF16 + LoRA + Clef capacity is unverified and may not fit')
    if check_model:
        try:
            from .storage import resolve_model
            result['model_path']=str(resolve_model(config))
        except Exception as exc:
            result['errors'].append(f'Model staging: {exc}')
    try:
        from .dataset import load_splits
        result['dataset_counts']={k:len(v) for k,v in load_splits(config).items()}
    except Exception as exc:
        result['errors'].append(f'Dataset: {exc}')
    for key in ['output_dir','report_dir']:
        try:
            path=Path(config['paths'][key]); path.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryFile(dir=path):
                pass
        except OSError as exc:
            result['errors'].append(f'Output permissions: {exc}')
    return result


def require_gpu(config: dict) -> dict:
    """Fail with all preflight errors before allocating model memory."""
    result=preflight(config)
    if result['errors']:
        raise RuntimeError('Preflight failed:\n'+'\n'.join(result['errors']))
    return result
