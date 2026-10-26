"""Record genuine v14 predictions for the authored offline demo; never train."""
import argparse,datetime,hashlib,json,math,time,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL='banking-streaming-context-v14-SYNTHETIC-DEVELOPMENT'

def validate_cases(cases):
 if len(cases)!=30 or len({x['id'] for x in cases})!=30:raise ValueError('Expected exactly 30 uniquely identified authored cases')
 for case in cases:
  if case['kind'] not in {'single','multi','voice'} or not case['context'].strip() or not case['response'].strip():raise ValueError('Invalid authored case')
 families={}
 for case in cases:
  if case['kind']=='voice':families.setdefault(case['family'],[]).append(case)
 for group in families.values():
  ordered=sorted(group,key=lambda x:x['stage'])
  if [x['stage'] for x in ordered]!=[1,2] or ordered[0]['context']!=ordered[1]['context'] or not ordered[1]['response'].startswith(ordered[0]['response']):raise ValueError('Invalid recorded voice-prefix family')

def capture(endpoint,output,checkpoint_sha,timeout=300):
 u=urllib.parse.urlparse(endpoint)
 if u.scheme!='http' or u.hostname not in {'127.0.0.1','localhost'} or u.username or u.password or u.path not in {'','/'} or u.query or u.fragment:raise ValueError('Use the private loopback endpoint origin')
 if len(checkpoint_sha)!=64 or any(c not in '0123456789abcdef' for c in checkpoint_sha):raise ValueError('Supply the verified checkpoint manifest SHA256')
 if output.exists():raise FileExistsError('Choose a new capture output; do not overwrite recorded results')
 cases_path=ROOT/'demo/cases.json';cases=json.loads(cases_path.read_text());validate_cases(cases)
 deadline=time.monotonic()+timeout
 while True:
  try:
   with urllib.request.urlopen(endpoint.rstrip('/')+'/health',timeout=5) as r:health=json.load(r)
   if health.get('ready'):break
  except (OSError,ValueError):pass
  if time.monotonic()>deadline:raise TimeoutError('Model not ready within the bounded check')
  time.sleep(3)
 if health.get('model_version')!=MODEL:raise ValueError('The connected model is not the verified v14 model')
 samples=[]
 for case in cases:
  body=json.dumps({'context':case['context'],'response':case['response']}).encode()
  request=urllib.request.Request(endpoint.rstrip('/')+'/predict',data=body,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=120) as r:result=json.load(r)
  p=result['prediction']
  if p.get('model_version')!=MODEL or list(p['decisions'])!=[f'D{i}' for i in range(1,8)]:raise ValueError('Model response contract changed')
  if not math.isfinite(result['latency_seconds']):raise ValueError('Invalid timing')
  for i in range(1,7):
   if type(p['decisions'][f'D{i}']['answer']) is not bool:raise ValueError('Expected native Boolean answer')
  samples.append({**case,'result':result,'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  print(json.dumps({'case_id':case['id'],'classification':p['classification']}),flush=True)
 data={'format':'v14_offline_prediction_capture_v1','case_count':30,'model_version':MODEL,'checkpoint_manifest_sha256':checkpoint_sha,'cases_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),'questions':health.get('questions',{}),'notice':'Genuine saved model outputs on authored synthetic demonstration cases. Not ground truth, independent evaluation or live inference. No outputs have been corrected.','samples':samples}
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 return data

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--endpoint',default='http://127.0.0.1:8788');p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-manifest-sha256',required=True);a=p.parse_args();capture(a.endpoint,a.output,a.checkpoint_manifest_sha256)
