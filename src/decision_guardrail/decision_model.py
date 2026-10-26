"""Verify Clef/LoRA/frozen-backbone invariants using actual parameters."""


def parameter_report(model, head_only: bool = False) -> dict:
    """Fail if architecture, precision, or trainability differs from the contract."""
    import torch
    if not getattr(model,'is_clef',False) or not hasattr(model,'head'):
        raise RuntimeError('Expected native Unsloth Clef head')
    report={'total':0,'trainable':0,'frozen':0,'lora':0,'head':0,'backbone_dtypes':{}}
    for name,p in model.named_parameters():
        n=p.numel(); report['total']+=n
        report['trainable' if p.requires_grad else 'frozen']+=n
        if name.startswith('head.'):
            if not p.requires_grad or p.dtype!=torch.float32:
                raise RuntimeError(f'Clef must be trainable FP32: {name}')
            report['head']+=n
        elif 'lora_' in name:
            if p.requires_grad==head_only:
                raise RuntimeError(f'Unexpected adapter trainability: {name}')
            report['lora']+=n
        else:
            if p.requires_grad:
                raise RuntimeError(f'Original backbone unexpectedly trainable: {name}')
            if p.is_floating_point():
                dtype=str(p.dtype); report['backbone_dtypes'][dtype]=report['backbone_dtypes'].get(dtype,0)+n
                # Native normalization/scalar parameters may remain FP32.
                if p.ndim>=2 and p.dtype!=torch.bfloat16:
                    raise RuntimeError(f'Backbone matrix not BF16: {name} {p.dtype}')
    if not report['head'] or not report['lora']:
        raise RuntimeError('Missing Clef or LoRA parameters')
    return report
