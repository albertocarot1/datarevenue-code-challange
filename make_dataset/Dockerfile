FROM python:3.8-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install curl -qy

# Install production dependencies.
RUN pip install --upgrade setuptools pip
ENV POETRY_VERSION 1.1.6
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.8
RUN exec bash && source $HOME/.poetry/env
ENV PATH $PATH:/root/.poetry/bin

# Copy local code to container image
COPY . /opt/make_dataset/

# Copy poetry files for docker caching (python environment is re-installed only if they change
COPY pyproject.toml poetry.lock /opt/make_dataset/
WORKDIR /opt/make_dataset/

RUN poetry config virtualenvs.create false \
                && poetry export --without-hashes -f requirements.txt --dev \
                |  poetry run pip install -r /dev/stdin \
                && poetry debug

RUN poetry install -v --no-root -n
