"""Shared command line for preflight, training, evaluation and inference."""
import argparse
import json
import logging
from pathlib import Path
import sys
from .config import load_config, merge, validate, assign
from .environment import set_offline
from .utils import read_json, write_json


def parser() -> argparse.ArgumentParser:
    """Create a consistent command parser."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare-data','preflight','train','evaluate','predict','verify-checkpoint'])
    p.add_argument('--platform',default='local',choices=['local','runpod','sagemaker'])
    p.add_argument('--compute',default='a40_48gb')
    p.add_argument('--experiment',default='streaming_context_v14')
    p.add_argument('--config',help='Additional YAML overlay after the four profile layers')
    p.add_argument('--resolved-config',help='Exact resolved JSON configuration, with environment and --set overrides')
    p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    p.add_argument('--checkpoint')
    p.add_argument('--context')
    p.add_argument('--response')
    p.add_argument('--full-model-smoke',action='store_true',help='Explicitly allocate GPU model weights after preflight')
    p.add_argument('--skip-model',action='store_true',help='Lightweight package/data checks only; no S3 staging')
    p.add_argument('--reload',action='store_true',help='Load checkpoint and compare saved probability fixtures')
    p.add_argument('--split',choices=['validation','test'],default='test')
    return p


def command_config(args) -> dict:
    """Resolve identical profile/environment/CLI handling for every command."""
    if not args.resolved_config and args.checkpoint and (Path(args.checkpoint)/'resolved_config.json').is_file():
        args.resolved_config=str(Path(args.checkpoint)/'resolved_config.json')
    if not args.resolved_config:
        return load_config(args.platform,args.compute,args.experiment,args.set,args.config)
    import yaml
    from .config import ENV_PATHS
    import os
    config=read_json(Path(args.resolved_config))
    if args.config:
        config=merge(config,yaml.safe_load(Path(args.config).read_text()))
    for env,key in ENV_PATHS.items():
        if os.environ.get(env):
            assign(config,key,os.environ[env])
    for item in args.set:
        key,sep,value=item.partition('=')
        if not sep:
            raise ValueError('Overrides use KEY=VALUE')
        assign(config,key,yaml.safe_load(value))
    validate(config)
    return config


def execute(args) -> dict:
    """Dispatch a command without loading GPU libraries for CPU operations."""
    config=command_config(args); set_offline()
    logging.basicConfig(level=config['logging']['level'],format='%(asctime)s %(levelname)s %(name)s %(message)s')
    if args.command=='prepare-data':
        from .preparation import prepare_dataset
        return prepare_dataset(config)
    if args.command=='preflight':
        from .environment import preflight
        report=preflight(config,not args.skip_model)
        if args.full_model_smoke and not report['errors']:
            from .model_loader import load_model
            from .decision_model import parameter_report
            model,_=load_model(config); report['parameters']=parameter_report(model,config['training']['head_only'])
        return report
    if args.command=='train':
        from .trainer import train
        return train(config)
    if not args.checkpoint:
        raise ValueError('--checkpoint is required')
    from .checkpoint import verify_checkpoint
    verify_checkpoint(args.checkpoint,config)
    if args.command=='verify-checkpoint':
        result={'checkpoint':args.checkpoint,'integrity':'passed','gpu_reload':'not_requested'}
        if args.reload:
            from .inference import Predictor, compare_predictions, prediction_decisions
            predictor=Predictor(config,args.checkpoint)
            fixtures=read_json(Path(args.checkpoint)/'reload_fixtures.json')
            if not fixtures:
                raise ValueError('No saved probability fixtures')
            max_delta=0.
            for fixture in fixtures:
                actual = predictor.predict_state(fixture['state']) if 'state' in fixture else predictor.predict(fixture['context'],fixture['response'])
                expected=fixture['prediction']
                max_delta=max(max_delta,compare_predictions(actual,expected))
            result.update(gpu_reload='passed',maximum_probability_delta=max_delta,checked_questions=list(prediction_decisions(actual)))
            write_json(Path(config['paths']['report_dir'])/'reload-verification.json',result)
        return result
    from .inference import Predictor
    predictor=Predictor(config,args.checkpoint)
    if args.command=='predict':
        if args.context is None or args.response is None:
            raise ValueError('--context and --response are required')
        return predictor.predict(args.context,args.response)
    from .dataset import load_splits
    from .evaluation import evaluate
    return evaluate(predictor,load_splits(config)[args.split],Path(config['paths']['report_dir']),config['evaluation']['cohort']+'-'+args.split)


def main(argv: list[str] | None = None) -> int:
    """Print structured output and return nonzero on failed checks."""
    args=parser().parse_args(argv)
    try:
        result=execute(args)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 1 if result.get('errors') else 0
    except Exception as exc:
        logging.getLogger(__name__).error('%s: %s',type(exc).__name__,exc)
        return 1


if __name__=='__main__':
    sys.exit(main())
