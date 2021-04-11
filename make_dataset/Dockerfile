FROM drtools/dask:latest
COPY . /opt/make_dataset

USER root

WORKDIR /opt/make_dataset
RUN pip install -r /opt/make_dataset/requirements.txt
