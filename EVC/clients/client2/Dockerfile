FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install torchvision \
    && pip install matplotlib --ignore-installed certifi \
    && pip install paho-mqtt \
    && pip install scipy \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python \
    && rm -rf ~/.cache/pip

ENV GLOO_SOCKET_IFNAME=eth0

WORKDIR /work2
COPY train2 ./train2
COPY client.py ./
COPY ecci_sdk.py ./
COPY task_generator.py ./
