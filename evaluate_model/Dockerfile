FROM debian:buster-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt-get install -y mecab-ipadic-utf8

ENV HOME="/root"

WORKDIR $HOME
RUN apt-get install -y git
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.8-dev
RUN pyenv global 3.8-dev

COPY . /opt/evaluate_model/
RUN pip install -r /opt/evaluate_model/requirements.txt

WORKDIR /opt/evaluate_model/
