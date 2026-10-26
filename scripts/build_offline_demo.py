"""Build self-contained offline prediction and v14 dataset-label pages."""
import argparse,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from capture_demo_predictions import validate_cases,MODEL
from decision_guardrail.inference import validate_answer
CLASSES=['factual_information','general_advice','personal_advice','other']
MASTER_SHA='dd682e928aa3b266e0f2fead96fcc810572b095a4d1d189229556a4b1d1a89c7'

def payload_json(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')

def write_page(path,title,payload):
 assets=ROOT/'demo/assets';page=(assets/'template.html').read_text(encoding='utf-8')
 for marker,value in [('__TITLE__',title),('__STYLE__',(assets/'style.css').read_text(encoding='utf-8')),('__PAYLOAD__',payload_json(payload)),('__SCRIPT__',(assets/'app.js').read_text(encoding='utf-8'))]:page=page.replace(marker,value)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(page,encoding='utf-8',newline='\n')

def build(output=None):
 output=Path(output or ROOT/'demo')
 cases_path=ROOT/'demo/cases.json';cases=json.loads(cases_path.read_text());validate_cases(cases)
 captured=json.loads((ROOT/'demo/predictions.json').read_text())
 if captured['model_version']!=MODEL or captured['case_count']!=30 or len(captured['samples'])!=30:raise ValueError('Incomplete or wrong-model capture')
 if captured['cases_sha256']!=hashlib.sha256(cases_path.read_bytes()).hexdigest():raise ValueError('Cases changed after capture')
 samples=captured['samples'];groups={}
 for i,(sample,case) in enumerate(zip(samples,cases,strict=True)):
  if {k:sample[k] for k in case}!=case:raise ValueError('Recorded result is not bound to the authored input')
  p=sample['result']['prediction']
  if p['model_version']!=MODEL or p['classification']!=p['labels']['D7'] or p['probabilities']!=p['decisions']['D7']['probabilities']:raise ValueError('Recorded model result contract changed')
  for q in [f'D{n}' for n in range(1,8)]:
   answer=validate_answer(p['decisions'][q],CLASSES if q=='D7' else ['true','false'],q!='D7')
   if p['labels'][q]!=answer['answer']:raise ValueError('Recorded native answers changed')
  if sample.get('family'):groups.setdefault(sample['family'],[]).append(i)
 prediction_payload={'mode':'predictions','rows':samples,'questions':captured['questions'],'trajectories':groups,'initial_index':next(i for i,r in enumerate(samples) if r['id']=='single-03'),'capture_provenance':{k:captured[k] for k in ['model_version','checkpoint_manifest_sha256','cases_sha256','notice']}}
 write_page(output/'index.html','V14 · Recorded model predictions',prediction_payload)
 master=ROOT/'data/generated/streaming_context_v14_release/records.jsonl'
 with master.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 if sha!=MASTER_SHA:raise ValueError('Frozen v14 master changed')
 records=[json.loads(x) for x in master.read_text(encoding='utf-8').splitlines()]
 contexts=[];lookup={};data_rows=[];source_groups={}
 for i,row in enumerate(records):
  context=row['input']['customer_context']
  if context not in lookup:lookup[context]=len(contexts);contexts.append(context)
  refs=row['metadata']['source_refs']
  for ref in refs:
   group=source_groups.setdefault(ref['scenario_id'],{})
   stage=ref['checkpoint_index']
   if stage in group and group[stage]!=i:raise ValueError('Conflicting source trajectory')
   group[stage]=i
  turns=re.findall(r'(?im)^(?:CUSTOMER|AGENT|ASSISTANT):',context)
  partial=any(not ref['is_final'] for ref in refs)
  kind='voice' if partial else 'multi' if len(turns)>1 else 'single'
  topic=re.sub(r'^v\d+[-_]','',row['metadata']['parent_topic']).replace('_',' ').replace('-',' ').capitalize()
  data_rows.append({'id':row['id'],'title':topic+' · '+row['id'],'kind':kind,'split':row['metadata']['split'],'context':lookup[context],'response':row['input']['agent_response'],'labels':row['labels']})
 trajectories={key:[value[stage] for stage in sorted(value)] for key,value in source_groups.items() if len(value)>1}
 for i,row in enumerate(records):
  available=[ref['scenario_id'] for ref in row['metadata']['source_refs'] if ref['scenario_id'] in trajectories]
  if available:data_rows[i]['family']=max(available,key=lambda key:len(trajectories[key]))
 for group in trajectories.values():
  for a,b in zip(group,group[1:]):
   if data_rows[a]['context']!=data_rows[b]['context'] or not data_rows[b]['response'].startswith(data_rows[a]['response']):raise ValueError('Non-extending saved dataset trajectory')
 qs=json.loads((master.parent/'native_questions.json').read_text())
 initial=next(i for i,r in enumerate(data_rows) if r['labels']['D7']=='personal_advice' and r['kind']=='multi')
 dataset_payload={'mode':'dataset','rows':data_rows,'contexts':contexts,'questions':qs,'trajectories':trajectories,'initial_index':initial,'master_sha256':MASTER_SHA}
 write_page(output/'dataset.html','V14 · Dataset label explorer',dataset_payload)
 return {'prediction_samples':len(samples),'dataset_records':len(records),'dataset_histories':len(contexts),'standalone_files':['index.html','dataset.html'],'dataset_master_unchanged':True}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();print(json.dumps(build(a.output),indent=2))
