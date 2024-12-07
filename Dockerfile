FROM python:3.10-slim
RUN apt-get update
RUN apt-get install -y g++ libopenblas-dev build-essential
RUN pip install pybind11

WORKDIR /python
COPY ./swish/ /python/swish
COPY ./Makefile /python/Makefile
RUN make swish

COPY ./setup.py /python/setup.py
COPY ./pyproject.toml /python/pyproject.toml
RUN pip install --no-cache setuptools build
RUN python3 -m build
RUN pip install /python/dist/Swish-*.whl

ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir="
RUN pip install tensorflow-cpu
RUN pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install omegaconf
RUN pip install hydra-core
RUN pip install pytorch_lightning
RUN pip install dvc

COPY ./.dvc/ /python/.dvc
COPY ./dvc_storage/ /python/dvc_storage
COPY ./data.dvc /python/data.dvc
COPY ./configs/ /python/configs
COPY ./dataset.py /python/dataset.py
COPY ./model.py /python/model.py
COPY ./train.py /python/train.py

RUN dvc pull

ENTRYPOINT ["python", "train.py"]