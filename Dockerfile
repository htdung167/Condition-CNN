FROM python:3.8 AS py3.8
FROM nvidia/cuda:11.2.0-runtime-ubuntu18.04
WORKDIR /ConditionCNN/
COPY --from=py3.8 / /
COPY requirements.txt /ConditionCNN/
RUN export PYTHONPATH=/ConditionCNN/
RUN pip install -r requirements.txt
RUN pip install wandb