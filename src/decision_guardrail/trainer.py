"""One training implementation for local, RunPod and SageMaker."""
from pathlib import Path
import subprocess
import sys
import time
from .environment import require_gpu, set_offline
from .dataset import load_splits, build_dataset, split_manifest
from .storage import resolve_model, base_identity
from .checkpoint import save_checkpoint, smoke_key
from .utils import read_json, write_json, fingerprint
from .schema import contract, contract_hash
from .progress import append_metric


def train(config: dict) -> dict:
    """Train native Clef + LoRA, calibrate, save and verify in a fresh process."""
    from .training_readiness import assert_training_ready
    assert_training_ready(config)
    set_offline(); environment=require_gpu(config)
    from unsloth import FastDecisionModel, DecisionTrainer
    import torch
    from transformers import TrainingArguments, TrainerCallback, set_seed
    from .model_loader import load_model
    from .decision_model import parameter_report
    from .calibration import calibrate
    from .inference import Predictor
    from .evaluation import evaluate
    set_seed(config['project']['seed'])
    splits=load_splits(config); base=resolve_model(config)
    identity=base_identity(base); key=smoke_key(config,identity)
    certificate=Path(config['model']['cache_dir'])/'smoke'/f'{key}.json'
    smoke=config['experiment']=='smoke_test'
    if not smoke and config['training']['require_smoke']:
        if not certificate.is_file() or read_json(certificate).get('status')!='passed':
            raise RuntimeError('Run the smoke_test with this base/compute/LoRA configuration first; no matching successful certificate exists')
    output=Path(config['paths']['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty paths.output_dir to preserve prior checkpoints')
    work=output.parent/(output.name+'-trainer')
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('Native trainer output exists; choose a fresh paths.output_dir')
    metric_path=Path(config['paths']['report_dir'])/'training_metrics.jsonl'
    if metric_path.exists():
        raise FileExistsError('Training metrics exist; choose a fresh paths.report_dir')
    # GPU loading begins only after all independent validation has passed.
    model,processor=load_model(config)
    token_coverage = {s:{} for s in splits}
    converted={s:build_dataset(rows,processor,model,token_coverage[s]) for s,rows in splits.items()}
    write_json(Path(config['paths']['report_dir'])/'token_coverage.json',token_coverage)
    t=config['training']
    interval=t.get('evaluation_interval')
    args=TrainingArguments(output_dir=str(work),per_device_train_batch_size=t['batch_size'],gradient_accumulation_steps=t['gradient_accumulation_steps'],num_train_epochs=t['epochs'],max_steps=t['max_steps'],learning_rate=t['learning_rate'],warmup_steps=t['warmup_steps'],weight_decay=t['weight_decay'],logging_steps=t['logging_steps'],save_steps=t['checkpoint_interval'],save_total_limit=2,report_to=[],bf16=True,fp16=False,seed=config['project']['seed'],dataloader_num_workers=t['dataloader_workers'],gradient_checkpointing=config['lora']['gradient_checkpointing'],eval_strategy='steps' if interval else 'epoch',eval_steps=interval,remove_unused_columns=False)
    observed={'head_gradient':False,'lora_gradient':False,'unexpected_gradient':False}
    handles=[]
    # Hooks observe real backward passes without retaining large gradients.
    def hook_for(group):
        def hook(gradient):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite {group} gradient')
            if torch.count_nonzero(gradient).item():
                observed[group+'_gradient']=True
            return gradient
        return hook
    before={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            group='head' if name.startswith('head.') else 'lora'
            handles.append(p.register_hook(hook_for(group)))
            # Snapshot one small vector per trainable tensor for actual update detection.
            before[name]=p.detach().flatten()[:64].float().cpu().clone()
    class FiniteLoss(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if logs and any(key in logs and not __import__('math').isfinite(logs[key]) for key in ('loss','eval_loss')):
                raise FloatingPointError('Non-finite training/validation loss')
            if logs:
                observed_logs = dict(logs, gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                                     gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                append_metric(metric_path,state.global_step,state.max_steps,state.epoch,observed_logs)
    from .validation_selection import make_callback
    selection=make_callback(TrainerCallback,t.get('validation_selection'))
    trainer=DecisionTrainer(model=model,tokenizer=processor,args=args,train_dataset=converted['train'],eval_dataset=converted['validation'],head_learning_rate=t['head_learning_rate'],callbacks=[FiniteLoss()]+([selection] if selection is not None else []))
    # Transformers normally filters NaN/Inf in logs; disable to expose failures.
    trainer.args.logging_nan_inf_filter=False
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    try:
        result=trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('GPU out of memory with unquantised BF16. Choose a larger GPU or reduce sequence length; no precision fallback was applied.') from exc
    finally:
        for handle in handles:
            handle.remove()
    duration=time.perf_counter()-start
    if selection is not None:selection.restore(model)
    selection_report=selection.report() if selection is not None else {'enabled':False}
    write_json(Path(config['paths']['report_dir'])/'validation_selection.json',selection_report)
    training_peak_bytes=torch.cuda.max_memory_allocated()
    updated={'head':False,'lora':False}
    for name,p in model.named_parameters():
        if name in before and not torch.equal(before[name],p.detach().flatten()[:64].float().cpu()):
            updated['head' if name.startswith('head.') else 'lora']=True
        if not p.requires_grad and p.grad is not None:
            observed['unexpected_gradient']=True
    if not __import__('math').isfinite(result.training_loss) or not observed['head_gradient'] or not updated['head'] or observed['unexpected_gradient'] or (not t['head_only'] and (not observed['lora_gradient'] or not updated['lora'])):
        raise RuntimeError(f'Training gradient/update validation failed: {observed}, {updated}')
    report=parameter_report(model,t['head_only'])
    calibration=calibrate(model,processor,splits,config)
    predictor=Predictor(config,model=model,processor=processor)
    # Validation is separate from calibration and untouched final test data.
    metrics=evaluate(predictor,splits['validation'],Path(config['paths']['report_dir']),'validation')
    fixtures=[]
    for row in splits['validation'][:3]:
        fixtures.append({'state':row['state'],'prediction':predictor.predict_state(row['state'])})
    # Fixtures can contain customer text, so only synthetic smoke fixtures are persisted.
    if not smoke:
        fixtures=[{'context':'What does interest mean?','response':'Interest is a charge for borrowing money.','prediction':predictor.predict('What does interest mean?','Interest is a charge for borrowing money.')}]
    metadata={'validation_selection':selection_report,'synthetic_development_only':bool(config['training'].get('development_run') or config['training'].get('fixture_smoke')=='v14_d1_d7_integration_20261026'),'independently_approved_training_records':0 if (config['training'].get('development_run') or config['training'].get('fixture_smoke')=='v14_d1_d7_integration_20261026') else len(splits['train']),'environment':environment,'config_sha256':fingerprint(config),'dataset_version':config['dataset']['version'],'schema':contract(config),'schema_sha256':contract_hash(config),'duration_seconds':duration,'gpu_peak_bytes':training_peak_bytes,'training':result.metrics,'token_coverage':token_coverage,'parameter_report':report,'gradient_checks':observed,'parameter_updates':updated,'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    checkpoint=save_checkpoint(model,processor,config,base,calibration,metadata,split_manifest(splits))
    write_json(checkpoint/'reload_fixtures.json',fixtures)
    # Publish fixture checksum too.
    manifest=read_json(checkpoint/'manifest.json')
    from .storage import file_hash
    manifest['reload_fixtures.json']=file_hash(checkpoint/'reload_fixtures.json'); write_json(checkpoint/'manifest.json',manifest)
    # Release all GPU references before a new process allocates the model.
    del trainer,predictor,model,processor,converted
    import gc
    gc.collect(); torch.cuda.empty_cache()
    command=[sys.executable,'-m','decision_guardrail.cli','verify-checkpoint','--checkpoint',str(checkpoint),'--resolved-config',str(checkpoint/'resolved_config.json'),'--reload']
    # Deliberately invalidate both saved machine paths in a disposable artifact.
    # The fresh process must recover using active config, not the saved references.
    from .checkpoint import stale_path_checkpoint
    with stale_path_checkpoint(checkpoint) as moved:
        command[command.index('--checkpoint')+1]=str(moved)
        subprocess.run(command,check=True)
    if smoke:
        certificate_data={'status':'passed','key':key,'checkpoint':str(checkpoint),'verified':'fresh-process offline probability comparison with nonexistent saved base paths'}
        write_json(certificate,certificate_data)
        write_json(checkpoint/'smoke_certificate.json',certificate_data)
        manifest=read_json(checkpoint/'manifest.json')
        manifest['smoke_certificate.json']=file_hash(checkpoint/'smoke_certificate.json')
        write_json(checkpoint/'manifest.json',manifest)
    return {'checkpoint':str(checkpoint),'training':result.metrics,'validation':metrics,'smoke_certificate':str(certificate) if smoke else None}
