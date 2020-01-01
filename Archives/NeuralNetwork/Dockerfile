# FROM tensorflow/tensorflow:1.12.0-gpu-py3
FROM tensorflow/tensorflow:1.12.0-py3
ENV ROOT=/test
RUN mkdir -p $ROOT
ADD . $ROOT
WORKDIR $ROOT
RUN pip install -r requirements_docker.txt