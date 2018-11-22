FROM anibali/pytorch:cuda-9.2
MAINTAINER Jerry Tworek

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
USER user

# Install some basic utilities
RUN sudo apt-get update && sudo apt-get install -y gcc g++ make cmake zlib1g-dev swig libsm6 libxext6 && sudo rm -rf /var/lib/apt/lists/*

RUN pip install -U pip cython
COPY --chown=user:user . /vel
WORKDIR /vel
RUN pip install -e .[gym,mongo]
RUN mv .velproject.dummy.yaml .velproject.yaml

CMD vel examples-configs/rl/atari/a2c/breakout_a2c.yaml train -d cuda:0
