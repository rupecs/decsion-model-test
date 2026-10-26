"""Render an AWS training request for human review. Never submit or upload."""
import argparse,json,re
from pathlib import Path

def request(*,name,image,role,instance,compute,base_s3,output_s3,subnets=None,security_groups=None,kms_key=None):
 if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?',name):raise ValueError('Invalid SageMaker job name')
 if not re.fullmatch(r'\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?/.+@sha256:[a-f0-9]{64}',image):raise ValueError('Use an ECR image pinned by SHA256 digest')
 if not re.fullmatch(r'arn:aws(?:-us-gov|-cn)?:iam::\d{12}:role/.+',role):raise ValueError('Use the approved training role ARN')
 if compute not in {'a40_48gb','l40s_48gb','a100_80gb'}:raise ValueError('Unsupported compute profile')
 if not re.fullmatch(r'ml\.[a-z0-9]+\.[a-z0-9]+',instance):raise ValueError('Supply a region-supported training instance type')
 for uri in [base_s3,output_s3]:
  if not re.fullmatch(r's3://[a-z0-9][a-z0-9.-]+/.+',uri):raise ValueError('Use S3 bucket/prefix URIs')
 if bool(subnets)!=bool(security_groups):raise ValueError('Subnets and security groups must be supplied together')
 job={'TrainingJobName':name,'RoleArn':role,'AlgorithmSpecification':{'TrainingImage':image,'TrainingInputMode':'File','ContainerArguments':['--compute',compute,'--authorize-synthetic-development']},'InputDataConfig':[{'ChannelName':'base_model','DataSource':{'S3DataSource':{'S3DataType':'S3Prefix','S3Uri':base_s3,'S3DataDistributionType':'FullyReplicated'}}}], 'OutputDataConfig':{'S3OutputPath':output_s3},'ResourceConfig':{'InstanceType':instance,'InstanceCount':1,'VolumeSizeInGB':200},'StoppingCondition':{'MaxRuntimeInSeconds':43200},'EnableManagedSpotTraining':False,'EnableNetworkIsolation':True,'Environment':{'DG_MODEL_PATH':'/opt/ml/input/data/base_model','DG_CACHE_DIR':'/tmp/decision-v14-cache','SM_MODEL_DIR':'/opt/ml/model','SM_OUTPUT_DATA_DIR':'/opt/ml/output/data','CUDA_VISIBLE_DEVICES':'0','DG_TRAINING_IMAGE':image}}
 if subnets:job['VpcConfig']={'Subnets':subnets,'SecurityGroupIds':security_groups}
 if kms_key:job['OutputDataConfig']['KmsKeyId']=kms_key
 return job

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['name','image','role','instance','compute','base-s3','output-s3']:p.add_argument('--'+name,required=True)
 p.add_argument('--subnets',nargs='+');p.add_argument('--security-groups',nargs='+');p.add_argument('--kms-key')
 p.add_argument('--authorize-synthetic-development',action='store_true');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if not a.authorize_synthetic_development:p.error('Explicit synthetic development opt-in required; independently approved records=0')
 job=request(name=a.name,image=a.image,role=a.role,instance=a.instance,compute=a.compute,base_s3=a.base_s3,output_s3=a.output_s3,subnets=a.subnets,security_groups=a.security_groups,kms_key=a.kms_key)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:f.write(json.dumps(job,indent=2)+'\n')
 print('Wrote request for review. No AWS API was called.')
