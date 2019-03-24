FROM nvidia/cuda:9.2-base-ubuntu16.04
MAINTAINER Jerry Tworek

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 \
    gcc g++ make cmake zlib1g-dev swig libsm6 libxext6 \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
	libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /userhome
WORKDIR /userhome

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /userhome
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Setup Python installation, we'll be using pyenv
ENV PYENV_ROOT="/userhome/.pyenv" PATH="/userhome/.pyenv/bin:/userhome/.pyenv/shims:$PATH"
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN pyenv update && \
 pyenv install 3.6.8 && \
 pyenv global 3.6.8 && \
 find $PYENV_ROOT/versions -type d '(' -name '__pycache__' -o -name 'test' -o -name 'tests' ')' -exec rm -rfv '{}' + && \
 find $PYENV_ROOT/versions -type f '(' -name '*.py[co]' -o -name '*.exe' ')' -exec rm -fv '{}' +

# Update pip and cython
RUN pip install -U pip cython

# Prepare vel directory
COPY --chown=user:user . /vel
WORKDIR /vel

# Install local installation of vel
RUN pip install -e .[gym,mongo,dev]
RUN mv .velproject.dummy.yaml .velproject.yaml

# Some default training command
CMD vel examples-configs/rl/atari/a2c/breakout_a2c.yaml train -d cuda:0
