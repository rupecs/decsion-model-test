from pathlib import Path
import importlib.util,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'sagemaker'))
from verify_v14 import verify
from sagemaker_train import main
from render_job import request

def test_packaged_dataset_and_native_joins_are_exact():
 result=verify()
 assert result['records']==17467 and result['splits']==dict(train=11883,validation=1640,calibration=2071,test=1873)
 assert result['independently_approved_records']==0

def test_sagemaker_entrypoint_requires_explicit_operator_opt_in():
 with pytest.raises(SystemExit) as error:main(['--compute','l40s_48gb','--check-config-only'])
 assert error.value.code==2

def test_sagemaker_cpu_config_check_never_loads_gpu():
 assert main(['train','--compute','l40s_48gb','--authorize-synthetic-development','--check-config-only'])==0
 assert 'torch' not in sys.modules and 'unsloth' not in sys.modules

def example():return dict(name='v14-development-test',image='111122223333.dkr.ecr.us-east-1.amazonaws.com/guardrail@sha256:'+'a'*64,role='arn:aws:iam::111122223333:role/ExampleTrainingRole',instance='ml.g6e.4xlarge',compute='l40s_48gb',base_s3='s3://example-private-bucket/licensed-base/',output_s3='s3://example-private-bucket/training-output/')

def test_sagemaker_request_is_bounded_and_uses_staged_base_only():
 job=request(**example())
 assert job['StoppingCondition']['MaxRuntimeInSeconds']==43200
 assert job['EnableManagedSpotTraining'] is False and job['EnableNetworkIsolation'] is True
 assert job['ResourceConfig']['InstanceCount']==1
 assert [x['ChannelName'] for x in job['InputDataConfig']]==['base_model']
 assert '--authorize-synthetic-development' in job['AlgorithmSpecification']['ContainerArguments']
 assert job['Environment']['SM_MODEL_DIR']=='/opt/ml/model'
 assert job['Environment']['SM_OUTPUT_DATA_DIR']=='/opt/ml/output/data'

@pytest.mark.parametrize('key,value',[('image','unversioned-image:latest'),('compute','l4_24gb'),('base_s3','https://example.com/model'),('name','bad job name'),('role','Administrator'),('subnets',['subnet-example'])])
def test_invalid_cloud_request_is_not_rendered(key,value):
 args=example();args[key]=value
 with pytest.raises(ValueError):request(**args)


def test_native_training_core_matches_the_tested_reference_source():
 import hashlib
 expected=json.loads((ROOT/'reports/reference_runtime_core_hashes.json').read_text())
 for name,sha in expected.items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha


def test_job_request_matches_the_installed_aws_service_schema_without_api_calls():
 import botocore.session
 from botocore.validate import validate_parameters
 model=botocore.session.get_session().get_service_model('sagemaker')
 validate_parameters(request(**example()),model.operation_model('CreateTrainingJob').input_shape)
