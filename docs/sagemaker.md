# Office AWS SageMaker runbook

This is a custom-container, **single visible GPU**, bounded synthetic-development training job. The Python code never provisions AWS resources. The reference run was on Runpod A40; **a SageMaker job has not been run as part of this handover**.

## 1. Confirm office prerequisites

Use the office-approved AWS account/region, training execution role, private S3 model/output prefixes, ECR repository, available SageMaker training instance quota, and required VPC/KMS settings. Resolve these with the operator; no real account identifiers or credentials belong in this public repository.

The operator must stage the **original licensed Gemma 4 E4B-IT snapshot**, including all shards, tokenizer and unified processor_config.json, at the root of the `base_model` S3 prefix. Do not add an extra nested snapshot directory unless DG_MODEL_PATH is deliberately adjusted. No model is downloaded from a repository during training. Confirm any provider licence/access requirements through the approved model acquisition process.

Pick a compute profile matching the GPU actually allocated:

- `l40s_48gb` for a single L40S 48GB; G6e is a possible hardware family, but confirm **SageMaker training** support, quota and price in the office region.
- `a100_80gb` only for an actual A100 80GB allocation. Many such instances have multiple GPUs: exposing one still bills the full instance.
- `a40_48gb` records the reference hardware and is not a claim that AWS offers an A40 instance.

Do not silently select a 24GB GPU or a multi-GPU instance. The entrypoint verifies actual CUDA GPU identity and BF16 support. Fresh coverage/smoke/reload is mandatory on whichever approved GPU is selected.

## 2. Build and publish an approved image

The provided Dockerfile pins the registry-resolved Linux/amd64 digest of the reference image tag. The original container runtime digest was not independently attested. The office may require an approved equivalent image; a different image must still satisfy the pinned runtime and pass actual GPU smoke. Do not claim an image build passed unless it was run.

```sh
# Run only in an approved build environment. Replace ECR_URI with the office URI.
docker build --platform linux/amd64 -f sagemaker/Dockerfile -t ECR_URI:v14 .
# Authenticate/push using the office-approved ECR workflow.
# Resolve the pushed image's actual digest and use ECR_URI@sha256:DIGEST below.
```

The Dockerfile creates a venv inheriting system CUDA PyTorch and installs this project against tested_gpu_constraints.txt. GPU dependencies are installed during image build, not through runtime internet access. No base model or private job configuration enters the image/build context. `.dockerignore` excludes weights, credentials, outputs and local environments.

The exact **dataset and 20-row smoke fixture are baked into the image** at /opt/program. This design preserves their manifest/native-file path bindings and avoids partial or accidentally mixed SageMaker dataset channels. The only input channel is `base_model`. Do **not** set SM_CHANNEL_TRAIN/VALIDATION/CALIBRATION/TEST to other data in this workflow; path/hash verification will refuse substitutions.

## 3. Render a request without submitting

The example values below are placeholders, not real account data. Replace every one and choose an instance confirmed supported for training in your region:

```sh
python sagemaker/render_job.py \
  --name v14-dev-UNIQUE \
  --image 111122223333.dkr.ecr.us-east-1.amazonaws.com/guardrail@sha256:REPLACE_WITH_64_HEX_DIGEST \
  --role arn:aws:iam::111122223333:role/REPLACE_WITH_APPROVED_TRAINING_ROLE \
  --instance REPLACE_WITH_REGION_SUPPORTED_ML_INSTANCE \
  --compute l40s_48gb \
  --base-s3 s3://example-private-bucket/licensed-base/ \
  --output-s3 s3://example-private-bucket/training-output/ \
  --authorize-synthetic-development \
  --output .local/v14-job.json
```

The renderer rejects unpinned images, missing synthetic opt-in, invalid basic identifiers and incomplete VPC parameters. It writes a new JSON file and calls **no AWS API**. Optional `--subnets`, `--security-groups` and `--kms-key` let the operator supply office settings. Review the complete request before submission.

Defaults: one instance, 200GB training volume, on-demand (no spot/resume assumption), 12-hour MaxRuntime and network isolation. IAM must grant the execution role only the necessary input S3 read, output S3 write, ECR pull, logs and any required KMS access; the submitting operator needs the appropriate SageMaker and PassRole rights. Apply office policy rather than copying administrator permissions.

## 4. Submit only after operator authorisation

```sh
aws sagemaker create-training-job --region OFFICE_REGION \
  --cli-input-json file://.local/v14-job.json
```

This command is billable. It is deliberately an explicit operator step, not part of any build, verification or test. Keep the job name, image digest, account/region, chosen GPU/price and request privately with the office run record.

SageMaker runs the container entrypoint, which handles an optional leading `train` argument and performs: CPU verification -> all-row official token coverage -> three-step smoke -> exact FP32/all-seven reload verification -> capped training/early stopping -> calibration/save/reload -> one held-out test. Any failed stage exits nonzero and blocks later stages. One-shot stage files prevent accidental duplicate execution.

## 5. Monitor, stop and retrieve

```sh
aws sagemaker describe-training-job --region OFFICE_REGION \
  --training-job-name JOB_NAME \
  --query '{Status:TrainingJobStatus,Failure:FailureReason,Artifacts:ModelArtifacts}'

# If explicitly stopping the job:
aws sagemaker stop-training-job --region OFFICE_REGION --training-job-name JOB_NAME
```

Inspect CloudWatch and saved stage logs rather than treating missing progress output as failure by itself. No automatic retry/resume is implemented. A SIGTERM or runtime ceiling can leave useful reports but no complete portable model. Use a new job/output after diagnosis, not a silent retry.

| Container path | Contents / SageMaker handling |
|---|---|
| /opt/ml/input/data/base_model | Original model, staged by the File channel; never overwritten |
| /opt/ml/model | Final adapter, exact FP32 head, calibration, processor, schema, split provenance and metadata; exported as model.tar.gz |
| /opt/ml/output/data | Coverage, smoke verification/certificate/metadata, stage logs, freeze in checkpoint metadata, phase/exit and final evaluation reports; exported as output.tar.gz |
| Temporary smoke/trainer/cache paths | Execution intermediates, not final portable model weights |

The smoke's duplicate adapter weights are temporary. The final model omits the original base and intermediate optimiser/trainer weights. Full package freeze, GPU/API environment, dataset/schema/base hashes and selection evidence are recorded in model metadata/reports.

Wait for the job's terminal status (`Completed`, `Failed` or `Stopped`) and verify the reported state. Managed training compute is released at job termination; S3/ECR output and model storage continue separately. This workflow creates no hosting endpoint. Do not leave an unrelated notebook, instance or endpoint running under the assumption that training completion stops it.

Download the returned model and output artifact S3 URIs into ignored local storage. Check all checkpoint manifest hashes. On the receiving GPU machine, set DG_MODEL_PATH to the same original base and run:

```sh
python -m decision_guardrail.cli verify-checkpoint \
  --checkpoint .local/model \
  --resolved-config .local/model/resolved_config.json --reload \
  --set paths.report_dir=.local/reload-report
```

This is real GPU work. It checks both saved base references through portable relocation and compares all seven fixture probabilities. Preserve calibration and keep the base unchanged. Reference certificates do not replace this actual-machine check.

## Primary AWS references

- [How SageMaker runs a custom training image](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)
- [Training filesystem and input channels](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)
- [Model and output artifact handling](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html)
- [CreateTrainingJob CLI contract](https://docs.aws.amazon.com/cli/latest/reference/sagemaker/create-training-job.html)
- [AWS accelerated hardware specifications](https://aws.amazon.com/ec2/instance-types/accelerated-computing/)

These references describe AWS behaviour. They do not establish this account's regional quota, permissions, price or completed GPU validation.

## Packaging boundary

Use this source checkout and Dockerfile; a wheel alone does not bundle the roughly 181-MiB frozen dataset/configuration. The coverage stage compares the base channel’s full config/shard identity with reports/reference_base_identity.json before GPU model loading. The workflow also copies source/development manifests and the supplied ECR digest into output provenance.
