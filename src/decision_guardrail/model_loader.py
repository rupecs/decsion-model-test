"""Original local BF16 Gemma with a new FP32 Clef head and LoRA adapters."""
import logging
from .config import validate
from .environment import configure_runtime, inspect_api
from .storage import resolve_model
from .decision_model import parameter_report


def load_model(config: dict, checkpoint: str | None = None):
    """Load native Unsloth model; never download, merge, quantise or change precision."""
    validate(config); configure_runtime(config)
    base=resolve_model(config)
    inspect_api()
    from unsloth import FastDecisionModel
    import torch
    kwargs={'dtype':torch.bfloat16,'load_in_4bit':False,'load_in_8bit':False,'full_finetuning':False,'local_files_only':True,'max_seq_length':config['model']['max_seq_length'],'random_state':config['project']['seed'],'use_gradient_checkpointing':'unsloth' if config['lora']['gradient_checkpointing'] else False}
    if checkpoint:
        from .checkpoint import relocated_checkpoint
        with relocated_checkpoint(checkpoint,base) as relocated:
            model,processor=FastDecisionModel.from_pretrained(model_name=str(relocated),**kwargs)
    else:
        model,processor=FastDecisionModel.from_pretrained(model_name=str(base),decision_head='clef',head_init=None,**kwargs)
        l=config['lora']
        # Native Clef path restricts these adapters to language layers.
        model=FastDecisionModel.get_peft_model(model,r=l['rank'],lora_alpha=l['alpha'],lora_dropout=l['dropout'],target_modules=l['target_modules'],bias='none',random_state=config['project']['seed'],use_gradient_checkpointing=kwargs['use_gradient_checkpointing'])
    if config['training']['head_only']:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    report=parameter_report(model,config['training']['head_only'])
    logging.getLogger(__name__).info('Model: Gemma 4 E4B-IT; pretrained backbone frozen; LoRA %s; Clef trainable; parameters: %s','frozen' if config['training']['head_only'] else 'trainable',report)
    return model,processor
