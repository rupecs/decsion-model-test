# Offline execution and data handling

Install dependencies and stage the original licensed base before entering the restricted training runtime. GPU imports are lazy; set_offline/configure_runtime runs before Unsloth/Transformers/PEFT. The model loader uses local_files_only=True and permits only the original local/S3 staging contract, without model repository downloads.

The SageMaker custom image contains v14 data and code; the managed File input channel stages the base. Network isolation is enabled in the rendered request. Use office-managed S3/ECR/KMS/VPC permissions; do not insert credentials into the image, repository, logs or job arguments.

CPU-only verification and replay do not import Torch or Unsloth. Native encoding, smoke, training and reload require the actual staged model and an explicitly authorised GPU run. A successful CPU test is never a GPU certificate.

Synthetic data in this repository is public research material. Normal inference submissions should still remain private: do not enable input logging or send text to feedback/telemetry services. A future real-customer corpus requires its own access, retention, publication and annotation decisions; do not add it to this public dataset directory.
