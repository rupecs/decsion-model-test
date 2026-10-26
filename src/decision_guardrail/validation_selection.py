"""Early stopping with exact CPU best-state retention, independent of HF saving.

Only trainable tensors and head buffers are copied; frozen base weights are never
cloned. Native exact portable saving/calibration occur after restoration.
"""
import math


def make_callback(base,settings):
    if not settings or settings.get('enabled') is not True:return None
    if (type(settings.get('patience')) is not int or settings['patience']<1 or
        type(settings.get('min_delta')) not in (int,float) or not math.isfinite(settings['min_delta']) or
        settings['min_delta']<0 or settings.get('restore_best') is not True):raise ValueError('Invalid validation selection settings')
    class Selection(base):
        def __init__(self):
            self.best_loss=None;self.best_step=None;self.anchor=None;self.bad=0;self.evaluations=0
            self.parameters=None;self.buffers=None;self.restored=False;self.stop_requested=False
        def on_evaluate(self,args,state,control,metrics=None,model=None,**kwargs):
            if metrics is None or 'eval_loss' not in metrics or model is None:raise RuntimeError('Validation loss/model missing')
            loss=float(metrics['eval_loss'])
            if not math.isfinite(loss):raise FloatingPointError('Non-finite validation loss')
            self.evaluations+=1
            if self.best_loss is None or loss<self.best_loss:
                self.best_loss=loss;self.best_step=state.global_step
                self.parameters={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
                self.buffers={name:b.detach().cpu().clone() for name,b in model.named_buffers() if name.startswith('head.')}
                if not self.parameters:raise RuntimeError('No trainable parameters to preserve')
            if self.anchor is None or loss<self.anchor-settings['min_delta']:
                self.anchor=loss;self.bad=0
            else:self.bad+=1
            if self.bad>=settings['patience']:
                self.stop_requested=True;control.should_training_stop=True
            return control
        def restore(self,model):
            if self.parameters is None:raise RuntimeError('No validation checkpoint was observed; refusing final-model fallback')
            import torch
            parameters=dict(model.named_parameters());buffers=dict(model.named_buffers())
            with torch.no_grad():
                for name,value in self.parameters.items():
                    target=parameters[name]
                    if target.dtype!=value.dtype or target.shape!=value.shape:raise RuntimeError('Best-state dtype/shape changed: '+name)
                    target.copy_(value.to(device=target.device))
                for name,value in self.buffers.items():
                    target=buffers[name]
                    if target.dtype!=value.dtype or target.shape!=value.shape:raise RuntimeError('Head-buffer dtype/shape changed: '+name)
                    target.copy_(value.to(device=target.device))
            self.restored=True
        def report(self):
            return dict(monitor='validation_loss',patience=settings['patience'],min_delta=settings['min_delta'],
                evaluations=self.evaluations,best_step=self.best_step,best_loss=self.best_loss,
                stop_requested=self.stop_requested,best_state_restored=self.restored,
                trainable_tensors=len(self.parameters or {}),head_buffers=len(self.buffers or {}),
                frozen_base_copied=False,state_precision='Original tensor dtype retained on CPU; no conversion')
    return Selection()
