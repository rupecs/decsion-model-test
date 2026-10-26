# Reproduce the tested Linux/amd64 software baseline, or supply an approved equivalent.
ARG BASE_IMAGE=runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851
FROM ${BASE_IMAGE}
WORKDIR /opt/program
COPY . /opt/program
RUN python3 -m venv --system-site-packages /opt/v14-venv \
 && /opt/v14-venv/bin/python -m pip install -e '.[gpu]' -c tested_gpu_constraints.txt \
 && /opt/v14-venv/bin/python -m pip check
ENV PATH="/opt/v14-venv/bin:${PATH}" PYTHONPATH=/opt/program/src PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
ENTRYPOINT ["/opt/v14-venv/bin/python", "/opt/program/scripts/sagemaker_train.py"]
