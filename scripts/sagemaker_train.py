"""One bounded SageMaker development job: coverage -> smoke -> train -> test.

This entrypoint never calls AWS or provisions infrastructure. The operator must
explicitly opt into synthetic development labels. No automatic restart/resume.
"""
import argparse,json,os,signal,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 if args and args[0]=='train':args=args[1:]
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--compute',required=True,choices=['a40_48gb','l40s_48gb','a100_80gb'])
 p.add_argument('--authorize-synthetic-development',action='store_true')
 p.add_argument('--check-config-only',action='store_true',help='CPU-only gate check; does not load model or allocate GPU')
 a=p.parse_args(args)
 if not a.authorize_synthetic_development:p.error('Explicit --authorize-synthetic-development is required. Zero labels are independently approved.')
 os.environ['DG_V14_SYNTHETIC_DEVELOPMENT']='v14_synthetic_development_20261026'
 os.environ.setdefault('DG_MODEL_PATH','/opt/ml/input/data/base_model')
 os.environ.setdefault('DG_CACHE_DIR','/tmp/decision-v14-cache')
 os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
 os.environ['PYTHONPATH']=str(ROOT/'src')
 from decision_guardrail.environment import set_offline
 set_offline()
 from decision_guardrail.config import load_config
 from decision_guardrail.training_readiness import assert_training_ready
 for profile in ['streaming_context_v14_development','smoke_context_v14_development']:
  assert_training_ready(load_config('sagemaker',a.compute,profile))
 if a.check_config_only:
  print(json.dumps({'gate':'passed_for_explicit_synthetic_development_only','independently_approved_records':0,'GPU_loaded':False}));return 0
 def interrupted(signum,frame):raise SystemExit('Training terminated by signal '+str(signum))
 signal.signal(signal.SIGTERM,interrupted)
 output=Path(os.environ.get('SM_OUTPUT_DATA_DIR','/opt/ml/output/data'))
 model=Path(os.environ.get('SM_MODEL_DIR','/opt/ml/model'))
 output.mkdir(parents=True,exist_ok=True);model.mkdir(parents=True,exist_ok=True)
 fd=os.open(output/'run.launch',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
 import shutil
 for name in ['package_manifest.json','development_pins.json']:
  if (ROOT/name).exists():shutil.copy2(ROOT/name,output/name)
 started=time.time();deadline=started+43100
 (output/'run_environment.json').write_text(json.dumps({'platform':'sagemaker','compute_profile':a.compute,'started_epoch':started,'training_image':os.environ.get('DG_TRAINING_IMAGE','not_supplied'),'max_workflow_seconds':43100,'max_optimizer_steps':2972,'independently_approved_records':0},indent=2)+'\n')
 def phase(name):
  (output/'phase.json').write_text(json.dumps({'phase':name,'timestamp':time.time()}))
 def execute(name,argv):
  phase(name)
  with (output/(name+'.log')).open('x') as log:
   child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:status=child.wait(timeout=max(1,deadline-time.time()))
   except BaseException:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
   if status:raise RuntimeError(f'{name} failed with exit {status}; inspect its saved log. No restart was attempted.')
 try:
  execute('cpu_data_verification',[sys.executable,'scripts/verify_v14.py'])
  execute('native_coverage',[sys.executable,'scripts/native_coverage.py',a.compute,str(output)])
  def training(profile,cp,report):
   return [sys.executable,'-m','decision_guardrail.cli','train','--platform','sagemaker','--compute',a.compute,'--experiment',profile,'--set',f'paths.output_dir={cp}','--set',f'paths.report_dir={report}']
  with tempfile.TemporaryDirectory(prefix='v14-smoke-') as scratch:
   smoke=Path(scratch)/'checkpoint';smoke_report=output/'smoke'
   execute('smoke',training('smoke_context_v14_development',smoke,smoke_report))
   execute('smoke_verification',[sys.executable,'scripts/verify_model_artifact.py',str(smoke),str(smoke_report),str(output/'smoke_verified.json')])
   import shutil
   for name in ['smoke_certificate.json','metadata.json']:
    shutil.copy2(smoke/name,output/('smoke_'+name))
  fd=os.open(output/'full_training.launch',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
  execute('training',training('streaming_context_v14_development',model,output/'reports'))
  execute('final_verification',[sys.executable,'scripts/verify_model_artifact.py',str(model),str(output/'reports'),str(output/'model_verified.json')])
  fd=os.open(output/'heldout_test.launch',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
  execute('heldout_test',[sys.executable,'-m','decision_guardrail.cli','evaluate','--checkpoint',str(model),'--resolved-config',str(model/'resolved_config.json'),'--split','test','--set','evaluation.cohort=v14-sagemaker-synthetic-development'])
  phase('completed');(output/'exit').write_text('0\n');return 0
 except BaseException:
  phase('failed');(output/'exit').write_text('1\n')
  text=traceback.format_exc();(output/'failure.txt').write_text(text)
  failure=output.parent/'failure';failure.write_text(text[-4000:])
  raise

if __name__=='__main__':raise SystemExit(main())
