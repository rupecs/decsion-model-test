"""Layered configuration shared by every entry point."""
import copy
import os
import sys
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT/'config/base.yaml').exists():
    ROOT = Path(sys.prefix)/'share/decision-guardrail'
ENV_PATHS = {
    'DG_MODEL_PATH': 'model.local_path', 'DG_MODEL_S3_URI': 'model.s3_uri',
    'DG_CACHE_DIR': 'model.cache_dir', 'DG_OUTPUT_DIR': 'paths.output_dir',
    'DG_DATASET_PATH': 'dataset.path', 'AWS_REGION': 'storage.region',
    'DG_S3_BUCKET': 'storage.bucket', 'DG_POLICY_PATH': 'policy.path',
}


def merge(base: dict, overlay: dict) -> dict:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def assign(config: dict, dotted: str, value: Any) -> None:
    """Assign a dotted configuration key."""
    keys = dotted.split('.')
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def validate(config: dict) -> None:
    """Reject unsafe architecture changes before importing GPU libraries."""
    m, h, t = config['model'], config['decision_head'], config['training']
    if config['platform'] not in {'local', 'runpod', 'sagemaker'}:
        raise ValueError('Unknown platform')
    if config['compute_profile'] not in {'a40_48gb','l4_24gb','l40s_48gb','a100_80gb'}:
        raise ValueError('Unknown compute profile')
    if m['architecture'] != 'gemma-4-E4B-it' or m['dtype'] != 'bfloat16':
        raise ValueError('Only original Gemma 4 E4B-IT in BF16 is supported')
    if not isinstance(m.get('compile', True), bool):
        raise ValueError('model.compile must be Boolean')
    if m.get('full_finetuning') or m.get('load_in_4bit') or m.get('load_in_8bit') or not m['local_files_only']:
        raise ValueError('Frozen, unquantised, offline backbone required')
    if h['architecture'] != 'clef' or h['dtype'] != 'float32' or h['initialization'] != 'random':
        raise ValueError('New native FP32 Clef head required')
    if m['source'] not in {'local','s3'}:
        raise ValueError('Model source must be local or s3')
    if not m.get('local_path') and m['source'] == 'local':
        raise ValueError('Set DG_MODEL_PATH or model.local_path')
    if m['source'] == 's3' and not str(m.get('s3_uri','')).startswith('s3://'):
        raise ValueError('Set model.s3_uri for S3 source')
    for key in ['batch_size','gradient_accumulation_steps','epochs','learning_rate','logging_steps','checkpoint_interval']:
        if isinstance(t[key], bool) or t[key] <= 0:
            raise ValueError(f'training.{key} must be positive')
    if m['max_seq_length'] < 32 or config['lora']['rank'] < 1:
        raise ValueError('Invalid sequence length or LoRA rank')
    if t['max_steps'] != -1 and t['max_steps'] < 1:
        raise ValueError('max_steps must be -1 or positive')
    interval = t.get('evaluation_interval')
    if interval is not None and (type(interval) is not int or interval < 1):
        raise ValueError('training.evaluation_interval must be a positive integer or null')
    if t['head_learning_rate'] is not None and t['head_learning_rate'] <= 0:
        raise ValueError('head_learning_rate must be positive or null')
    if config['experiment']=='smoke_test' and not 1 <= t['max_steps'] <= 3:
        raise ValueError('smoke_test requires 1–3 optimization steps')
    if config['evaluation']['bins'] < 1 or config['evaluation']['bootstrap_samples'] < 0:
        raise ValueError('Invalid evaluation bin/bootstrap count')
    labels = config['dataset']['labels']
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError('At least two unique labels required')
    if set(config['dataset']['splits']) != {'train','validation','calibration','test'}:
        raise ValueError('Four independent splits required')
    sizes = list(config['dataset']['splits'].values())
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or v <= 0 for v in sizes):
        raise ValueError('Split sizes must be positive')
    if not all(isinstance(v,int) for v in sizes) and abs(sum(sizes)-1) > 1e-6:
        raise ValueError('Fractional splits must sum to one')
    from .schema import validate_questions
    validate_questions(config)


def load_config(platform: str = 'local', compute: str = 'a40_48gb', experiment: str = 'streaming_context_v14', overrides: list[str] | None = None, config_file: str | None = None, root: Path = ROOT) -> dict:
    """Load base/platform/compute/experiment, environment, then explicit overrides."""
    for name in (platform,compute,experiment):
        if '/' in name or '..' in name:
            raise ValueError('Profile names cannot contain paths')
    config: dict = {}
    for path in [root/'config/base.yaml', root/f'config/platforms/{platform}.yaml', root/f'config/compute/{compute}.yaml', root/f'config/experiments/{experiment}.yaml']:
        config = merge(config, yaml.safe_load(path.read_text()) or {})
    if config_file:
        config = merge(config, yaml.safe_load(Path(config_file).read_text()) or {})
    for env, key in ENV_PATHS.items():
        if os.environ.get(env):
            assign(config, key, os.environ[env])
    if config['platform'] == 'sagemaker':
        for env, key in {'SM_MODEL_DIR':'paths.output_dir','SM_OUTPUT_DATA_DIR':'paths.report_dir'}.items():
            if os.environ.get(env):
                assign(config,key,os.environ[env])
        channels = {}
        for split in ['train','validation','calibration','test']:
            if os.environ.get('SM_CHANNEL_'+split.upper()):
                channels[split] = os.environ['SM_CHANNEL_'+split.upper()]
        if channels:
            # A partial channel set must never borrow the remaining files from a
            # local pilot profile. load_splits will require all four channels.
            config['dataset']['files'] = channels
    for override in overrides or []:
        key, sep, value = override.partition('=')
        if not sep:
            raise ValueError('Overrides use dotted.key=value')
        assign(config,key,yaml.safe_load(value))
    for section, keys in {'model':['local_path','cache_dir'],'paths':['output_dir','report_dir'],'dataset':['path','provenance_path','manifest_path'],'policy':['path']}.items():
        for key in keys:
            if config[section].get(key) and not str(config[section][key]).startswith('s3://'):
                p = Path(config[section][key]).expanduser()
                config[section][key] = str(p if p.is_absolute() else root/p)
    for split, source in config['dataset']['files'].items():
        if not str(source).startswith('s3://'):
            path = Path(source).expanduser()
            config['dataset']['files'][split] = str(path if path.is_absolute() else root/path)
    if config['dataset'].get('schema') == 'd1_d6_v1' and config['dataset'].get('questions') is None:
        config['dataset']['questions'] = yaml.safe_load((root/'config/dataset/question_schema_v1.yaml').read_text())
    if config['dataset'].get('schema') == 'banking_streaming_d1_d7_v4_binary_draft' and config['dataset'].get('questions') is None:
        config['dataset']['questions'] = yaml.safe_load((root/'config/dataset/question_schema_runtime_v4.yaml').read_text())
    if config['dataset'].get('schema') == 'banking_streaming_d1_d7_v5_context_draft' and config['dataset'].get('questions') is None:
        config['dataset']['questions'] = yaml.safe_load((root/'config/dataset/question_schema_runtime_v5.yaml').read_text())
    if config['dataset'].get('schema') == 'banking_streaming_d1_d7_v6_context_draft' and config['dataset'].get('questions') is None:
        config['dataset']['questions'] = yaml.safe_load((root/'config/dataset/question_schema_runtime_v6.yaml').read_text())
    validate(config)
    return config
