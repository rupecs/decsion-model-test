from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import json,sys
from types import SimpleNamespace
import pytest
from decision_guardrail.config import load_config
from decision_guardrail.dataset import load_splits
from decision_guardrail.training_readiness import training_readiness
from decision_guardrail.validation_selection import make_callback


@pytest.fixture(autouse=True)
def explicit_development_test_opt_in(monkeypatch):
    monkeypatch.setenv('DG_V14_SYNTHETIC_DEVELOPMENT','v14_synthetic_development_20261026')

def cfg(smoke=False):return load_config(platform='runpod',compute='a40_48gb',experiment='smoke_context_v14_development' if smoke else 'streaming_context_v14_development')


def test_exact_full_and_smoke_only_are_admitted_with_zero_approved_labels():
    for smoke,count in [(False,17467),(True,20)]:
        c=cfg(smoke);r=training_readiness(c)
        assert r['ready'] and r['approved_records']==0 and not r['production_ready']
        assert sum(map(len,load_splits(c).values()))==count
    original=load_config(experiment='streaming_context_v14')
    assert not training_readiness(original)['ready']


@pytest.mark.parametrize('smoke', [False,True])
@pytest.mark.parametrize('change', ['extra_steps','bool_steps','epochs','compile','precision','rank','accum','selection','unknown','data'])
def test_changes_cannot_expand_exception(smoke,change):
    c=cfg(smoke)
    if change=='extra_steps':c['training']['max_steps']+=1
    elif change=='bool_steps':c['training']['max_steps']=True
    elif change=='epochs':c['training']['epochs']+=1
    elif change=='compile':c['model']['compile']=True
    elif change=='precision':c['decision_head']['dtype']='bfloat16'
    elif change=='rank':c['lora']['rank']=32
    elif change=='accum':c['training']['gradient_accumulation_steps']=1
    elif change=='selection':c['training']['validation_selection']['restore_best']=False
    elif change=='unknown':c['training']['development_run']='other'
    elif change=='data':c['dataset']['files']['train']='somewhere_else.jsonl'
    assert not training_readiness(c)['ready']


class Tensor:
    def __init__(self,value,trainable=True,dtype='float32'):self.value=value;self.requires_grad=trainable;self.dtype=dtype;self.shape=(1,);self.device='cpu'
    def detach(self):return self
    def cpu(self):return self
    def clone(self):return Tensor(self.value,self.requires_grad,self.dtype)
    def to(self,**kwargs):return self
    def copy_(self,other):self.value=other.value
class Model:
    def __init__(self):self.p={'head.weight':Tensor(1),'lora.weight':Tensor(2,dtype='bfloat16'),'base.weight':Tensor(3,False,dtype='bfloat16')};self.b={'head.index':Tensor(4,False,'int64'),'base.cache':Tensor(5,False)}
    def named_parameters(self):return self.p.items()
    def named_buffers(self):return self.b.items()


def test_selection_stops_after_three_non_improving_checks_and_restores_exact_dtypes(monkeypatch):
    selection=make_callback(object,dict(enabled=True,patience=3,min_delta=.001,restore_best=True));model=Model();control=SimpleNamespace(should_training_stop=False)
    for step,loss in [(100,1.),(200,.9),(300,.91),(400,.92),(500,.93)]:
        model.p['head.weight'].value=step;model.p['lora.weight'].value=step+1;model.b['head.index'].value=step+2
        selection.on_evaluate(None,SimpleNamespace(global_step=step),control,metrics={'eval_loss':loss},model=model)
    assert control.should_training_stop and selection.best_step==200
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(no_grad=nullcontext))
    model.p['base.weight'].value=777;model.b['base.cache'].value=888
    selection.restore(model)
    assert model.p['head.weight'].value==200 and model.p['head.weight'].dtype=='float32'
    assert model.p['lora.weight'].value==201 and model.p['lora.weight'].dtype=='bfloat16'
    assert model.b['head.index'].value==202
    assert model.p['base.weight'].value==777 and model.b['base.cache'].value==888
    assert selection.report()['best_state_restored'] and not selection.report()['frozen_base_copied']


def test_true_minimum_weights_are_preserved_even_below_patience_delta():
    s=make_callback(object,dict(enabled=True,patience=3,min_delta=.01,restore_best=True));m=Model();c=SimpleNamespace(should_training_stop=False)
    for step,loss in [(1,1.),(2,.995),(3,.994),(4,.993)]:s.on_evaluate(None,SimpleNamespace(global_step=step),c,metrics={'eval_loss':loss},model=m)
    assert c.should_training_stop and s.best_step==4


def test_missing_and_nonfinite_validation_fail_closed():
    s=make_callback(object,dict(enabled=True,patience=3,min_delta=.001,restore_best=True));m=Model();c=SimpleNamespace(should_training_stop=False)
    with pytest.raises(RuntimeError,match='No validation'):s.restore(m)
    with pytest.raises(FloatingPointError):s.on_evaluate(None,SimpleNamespace(global_step=1),c,metrics={'eval_loss':float('nan')},model=m)
    with pytest.raises(RuntimeError,match='missing'):s.on_evaluate(None,SimpleNamespace(global_step=1),c,metrics={},model=m)


def test_no_opt_in_is_default_deny(monkeypatch):
    monkeypatch.delenv('DG_V14_SYNTHETIC_DEVELOPMENT',raising=False)
    assert not training_readiness(cfg())['ready']
    assert not training_readiness(cfg(True))['ready']

@pytest.mark.parametrize('compute',['a40_48gb','l40s_48gb','a100_80gb'])
def test_exact_same_data_on_explicit_actual_compute(compute):
    c=load_config(platform='sagemaker',compute=compute,experiment='streaming_context_v14_development')
    result=training_readiness(c)
    assert result['ready'] and result['approved_records']==0 and not result['production_ready']

def test_untested_small_profile_cannot_expand_contract():
    c=cfg();c['compute_profile']='l4_24gb'
    assert not training_readiness(c)['ready']
