# Vel 0.4

[![Build Status](https://travis-ci.org/MillionIntegrals/vel.svg?branch=master)](https://travis-ci.org/MillionIntegrals/vel)
[![PyPI version](https://badge.fury.io/py/vel.svg)](https://badge.fury.io/py/vel)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/MillionIntegrals/vel/blob/master/LICENSE)


Bring **velocity** to deep-learning research.


This project hosts a collection of **highly modular** deep learning components that are tested to be working well together.
A simple yaml-based system ties these modules declaratively using configuration files.


This is still an early version and a hobby project, so documentation is unfortunately nonexistent.
I've made an effort to make the code clear and provide many usage examples,
but whenever there was a tradeoff to be made between simplicity and modularity
I've chosen modularity first and simplicity second. Therefore, high emphasis is made on interfaces between components.


Having conducted a few research projects, I've gathered a small collection of repositories 
lying around with various model implementations suited to a particular usecase. 
Usually, starting a new project involved copying pieces of code from 
one or multiple of these past experiments, gluing, tweaking and debugging
them until the code started working in a new setting. 


After repeating that pattern multiple times, I've decided that this is the
time to bite the bullet and start organising deep learning models
into a structure that is designed to be reused rather than copied over.


As a goal, it should be enough to write a config file that
wires existing components together and defines their hyperparameters
for most common applications.
If that's not the case few bits of custom glue code should do the job.


This repository is still in an early stage of that journey but it will grow
as I'll be putting work into it.

For up-to-date benchmarks, look here:
[Benchmarks](docs/Benchmarks.md)


# Blogposts

- https://blog.millionintegrals.com/vel-pytorch-meets-baselines/


# How to run it 

Project can be installed from PyPi via `pip install vel` but also can be checked out from github
and installed directly by running
```bash
pip install -e .
```
from the repository root directory.

This project requires Python at least 3.6 and PyTorch 1.3.
If you want to run YAML config examples, you'll also need a **project configuration file**
`.velproject.yaml`. An example is included in this repository.

Default project configuration writes logs to the tensorboard directory `output/tensorboard`
under the main directory. Output modules to visdom, mongodb and wandb are also implemented.

If you don't want any logging, there is included another example file `.velproject.dummy.yaml`
that writes training progress to the standard output only.
To use it, just rename it to `.velproject.yaml`.


# Features

- Models should be runnable from the configuration files
  that are easy to store in version control, generate automatically and diff.
  Codebase should be generic and do not contain any of the model hyperparameters.
  Unless user intervenes, it should be obvious which model was run
  with which hyperparameters and what output it gave.
- The amount of "magic" in the framework should be limited and it should be easy to
  understand what exactly the model is doing for newcomers already comfortable with PyTorch. 
- All state-of-the-art models should be implemented in the framework with accuracy
  matching published results.
  For up-to-date benchmarks, look here: [Benchmarks](docs/Benchmarks.md)
- All common deep learning workflows should be fast to implement, while 
  uncommon ones should be possible, at least as far as PyTorch allows.
  
  
# Implemented models - Computer Vision

Several models are already implemented in the framework and have example config files
that are ready to run and easy to modify for other similar usecases:

- Residual networks (resnets) trained on Cifar10 dataset replicating published performance
- Cats vs dogs classification using transfer learning from a resnet34 model pretrained on 
  ImageNet
  
# Implemented models - Natural language processing

- Character-level language models based on LSTM and GRU recurrent networks, with example trained on
  works of Shakespeare
- Sentiment analysis of IMDB movie reviews
  
# Implemented models - Reinforcement learning

- Support for continuous and discrete environment action spaces
- Basic support for recurrent policies for A2C and PPO
- Following policy gradient reinforcement learning algorithms:
    - Advantage Actor-Critic (A2C)
    - Proximal Policy Optimization (PPO)
    - Trust Region Policy Optimization (TRPO)
    - Actor-Critic with Experience Replay (ACER)
    - Deep Deterministic Policy Gradient (DDPG)
- Deep Q-Learning (DQN) as described by DeepMind in their Nature publication with following 
  improvements:
    - Double DQN
    - Dueling DQN
    - Prioritized experience replay
    - N-Step Bellman updates
    - Distributional Q-Learning
    - Noisy Networks for Exploration
    - Rainbow (combination of the above)
    
# Implemented models - Unsupervised learning

- A simple AutoEncoder (AE) with example on MNIST dataset.
- Latent variable models:
    - Variational AutoEncoders (VAE)
    - Importance Weighted AutoEncoder (IWAE)
    - Vector-Quantised Variational AutoEncoder (VQ-VAE)


# Examples

Most of the examples for this framework are defined using config files in the
`examples-configs` directory with sane default hyperparameters already selected.

For example, to run the A2C algorithm on a Breakout atari environment, simply invoke:

```
python -m vel.launcher examples-configs/rl/atari/atari_a2c.yaml train
```

If you install the library locally, you'll have a special wrapper created
that will invoke the launcher for you. Then, above becomes:

```
vel examples-configs/rl/atari/atari_a2c.yaml train
```

General command line interface of the launcher is:

```
python -m vel.launcher CONFIGFILE COMMAND --device PYTORCH_DEVICE -r RUN_NUMBER -s SEED
```

Where `PYTORCH_DEVICE` is a valid name of pytorch device, most likely `cuda:0`.
Run number is a sequential number you wish to record your results with.

If you prefer to use the library from inside your scripts, take a look at the 
`examples-scripts` directory. From time to time I'll be putting some examples in there as
well. Scripts generally don't require any MongoDB or Visdom setup, so they can be run straight
away in any setup, but their output will be less rich and less informative.

# Docker

Dockerized version of this library is available in from the Docker Hub as
`millionintegrals/vel`. Link: https://hub.docker.com/r/millionintegrals/vel/

# PyPI

```
pip install vel
```

or

```
pip install vel[gym,mongo,visdom]
```

# Projects using Vel

- https://github.com/MillionIntegrals/baselines-experiments
- https://github.com/MillionIntegrals/vel-miniworld

# Glossary

For a glossary of terms used in the library please refer to [Glossary](docs/Glossary.md).
If there is anything you'd like to see there, feel free to open an issue or make a pull request.

# Bibliography

For a more or less exhaustive bibliography please refer to [Bibliography](docs/Bibliography.md).


# Roadmap

For each major version I'll try to keep master branch stable together with what's
currently published on PyPI. At the same time I'll proceed with implementing new
features on a release branch that will be merged after the testing is done and
a release is ready.

Below is a hypothetical set of features I somehow speculate to include in version
0.4 of Vel:

Very likely to be included:
- Neural machine translation using RNNs and Transformer Networks
- Soft actor-critic
- Twin Delayed DDPG


Possible to be included:
- Generative adversarial networks

For version 0.5 I'll again be looking to expand widely on the spectrum of available models in the framework,
as well as I'll try to support **multi-gpu** training by data parallelism.

Work in progress roadmap:

- Popart reward normalization
- PixelCNN
- MADE: Masked Autoencoder for Distribution Estimation
- Variational AutoEncoder with Inverse Autoregressive Flow


# Directories

Below I'll list brief explanation about contents of main top-level directories.

- `docs` - Few markdown documents about the framework
- `examples-configs` - Ready to run configs with tried and tested models, usually heavily inspired by existing
 literature.
- `examples-notebooks` - A few examples of how to interact with `vel` from the level of IPython notebook
- `vel` - Root for the Python source of the package
  - `vel.api` - Interfaces and base classes used all over the codebase. To be used in source code only and not 
    referenced from config files.
  - `vel.callback` - Definitions of callbacks that can be used in the training process. Can be referenced both by code
  and by the config files.
  - `vel.command` - Commands that can be used in your configuration files, and there isn't much need to refer to
  them from code.
  - `vel.data` - Various classes for handling data sources and data transformations. Referenced both by source code
   and config files.
  - `vel.function` - Interfaces for creating various functions/interpolators, to be refereced by config files.
  - `vel.internal` - Functions and classes to be only used by `vel` internally, and not by by user code nor configs.
  - `vel.metric` - Code for tracking metrics during training of your models. To be used by both code and configs.
  - `vel.model` - Definition of models, which is kind of an end-package that references all other packages. Models
  contain most other parts of the pipeline and define a training procedure.
  - `vel.module` - Various useful definitions of PyTorch modules, to be used when defining your own `models` and
  `layers`.
  - `vel.net` - "Network" module that may be referenced by a model to define neural network architecture used.
      - `vel.net.layer` - Modular layer system for defining networks declaratively in configuration files.
  - `vel.notebook` - Utilities for interfacing with `vel` using IPython notebooks
  - `vel.openai` - Imported parts of the codebase of `openai/baselines` that I didn't want to bring as a package
  dependency. To be referenced in code mostly.
  - `vel.optimizer` - Various implementations of deep learning optimizers. To be referenced mostly by scripts.
  - `vel.rl` - Meta package for everything related to Reinforcement Learning
    - `vel.rl.api` - Interfaces and base classes to be used for Reinforcement Learning models and other classes.
    - `vel.rl.buffer` - All classes relating to experience replay and experience buffers
    - `vel.rl.command` - Commands used for RL training
    - `vel.rl.env` - Basic reinforcement learning environments, mostly based on OpenAI gym
    - `vel.rl.env_roller` - Classes for generating environment rollouts
    - `vel.rl.layer` - Layers desined especially for RL
    - `vel.rl.module` - PyTorch modules designed for RL
    - `vel.rl.policy` - Equivalent of `vel.model` for RL
    - `vel.rl.reinforcer` - Reinforcer manages RL training, and corresponds to `Trainer` in Supervised Learning
    - `vel.rl.vecenv` - Utilities for vectorizing environments and stepping multiple environments at the same time
  - `vel.scheduler` - Classes helping to set up learning rate schedules for the optimizers. To be referenced mostly
  by scripts.
  - `vel.storage` - Everything about persisting models and metrics. To be referenced mostly by configuration files.
  - `vel.train` - Utilities for defining more generic training loops of models. To be referenced in both code and
  config.
  - `vel.util` - Collection of various utility functions to be used by all other modules.


# Citing

If you use `vel` in your research, you can cite it as follows:

```
@misc{tworek2018vel,
  author = {Jerry Tworek},
  title = {vel},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MillionIntegrals/vel}},
}
```


# Alternatives, similar projects

- https://github.com/NervanaSystems/coach
- https://github.com/google/dopamine
- https://github.com/openai/baselines
- https://github.com/rlpy/rlpy
- https://github.com/rlworkgroup/garage
- https://github.com/unixpickle/anyrl-py
- https://github.com/zuoxingdong/lagom
- https://github.com/inoryy/reaver-pysc2
