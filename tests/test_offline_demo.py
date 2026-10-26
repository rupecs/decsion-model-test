"""Check offline demos preserve source labels and genuine captured outputs."""
import hashlib,json,re,sys
from pathlib import Path
from collections import Counter
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_offline_demo import build,payload_json,MASTER_SHA
from capture_demo_predictions import validate_cases,MODEL,capture


def payload(name):
 page=(ROOT/'demo'/name).read_text()
 match=re.search(r'<script id="demo-data" type="application/json">(.*?)</script>',page,re.S)
 assert match
 return page,json.loads(match[1])


def test_prediction_gallery_uses_every_original_captured_result_and_input():
 cases=json.loads((ROOT/'demo/cases.json').read_text());validate_cases(cases)
 saved=json.loads((ROOT/'demo/predictions.json').read_text())
 _,display=payload('index.html')
 assert Counter(c['kind'] for c in cases)==dict(single=10,multi=10,voice=10)
 assert display['rows']==saved['samples']
 assert saved['cases_sha256']==hashlib.sha256((ROOT/'demo/cases.json').read_bytes()).hexdigest()
 assert saved['model_version']==MODEL
 assert saved['checkpoint_manifest_sha256']=='7e71e83eae65fab5787f04c9696dfc544223635f1c8e1ae31835d8c64b980b36'
 for original,recorded in zip(cases,display['rows'],strict=True):
  assert {k:recorded[k] for k in original}==original
 assert len(display['trajectories'])==5


def test_dataset_explorer_matches_all_frozen_inputs_labels_and_partitions():
 path=ROOT/'data/generated/streaming_context_v14_release/records.jsonl'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==MASTER_SHA
 records=[json.loads(x) for x in path.read_text().splitlines()]
 _,display=payload('dataset.html')
 assert len(display['rows'])==len(records)==17467
 for source,row in zip(records,display['rows'],strict=True):
  assert row['id']==source['id'] and row['labels']==source['labels']
  assert row['split']==source['metadata']['split']
  assert display['contexts'][row['context']]==source['input']['customer_context']
  assert row['response']==source['input']['agent_response']
  assert 'result' not in row and 'probabilities' not in row


def test_standalone_pages_disable_network_and_bind_every_saved_stage():
 for name in ['index.html','dataset.html']:
  page,data=payload(name)
  assert "connect-src 'none'" in page
  assert not re.search(r'<(?:script|link|img)\b[^>]*(?:src|href)=',page,re.I)
  for forbidden in ['fetch(', 'XMLHttpRequest', 'WebSocket(', 'sendBeacon(', 'eval(', 'innerHTML']:
   assert forbidden not in page
  for indices in data['trajectories'].values():
   for first,second in zip(indices,indices[1:]):
    a,b=data['rows'][first],data['rows'][second]
    assert a['context']==b['context'] and b['response'].startswith(a['response'])


def test_offline_build_is_reproducible_without_network(tmp_path,monkeypatch):
 import urllib.request
 def reject(*args,**kwargs):raise AssertionError('Offline build attempted networking')
 monkeypatch.setattr(urllib.request,'urlopen',reject)
 assert build(tmp_path)['dataset_master_unchanged']
 for name in ['index.html','dataset.html']:
  assert (tmp_path/name).read_bytes()==(ROOT/'demo'/name).read_bytes()


def test_script_payload_cannot_close_its_container():
 unsafe={'text':'</script><script>alert(1)</script> & <hello>'}
 encoded=payload_json(unsafe)
 assert '<' not in encoded and json.loads(encoded)==unsafe


@pytest.mark.parametrize('endpoint',['https://example.com','http://example.com','http://user:pass@localhost','http://127.0.0.1/path'])
def test_capture_refuses_public_or_credential_bearing_endpoints(endpoint,tmp_path):
 with pytest.raises(ValueError,match='loopback'):capture(endpoint,tmp_path/'new.json','a'*64)
