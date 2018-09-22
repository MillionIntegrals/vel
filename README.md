# Vel

[![Build Status](https://travis-ci.org/MillionIntegrals/vel.svg?branch=master)](https://travis-ci.org/MillionIntegrals/vel)
[![PyPI version](https://badge.fury.io/py/vel.svg)](https://badge.fury.io/py/vel)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/MillionIntegrals/vel/blob/master/LICENSE)


Bring **velocity** to deep-learning research,
by providing tried and tested large pool of prebuilt components that are 
known to be working well together.

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

# Requirements

This project requires Python 3.6 and PyTorch 0.4.1. Default project configuration writes
metrics to MongoDB instance open on localhost port 27017 and Visom instance 
on localhost port 8097. These can be changed in project-wide config file
`.velproject.yaml`.

# Features

- Models should be runnable from the configuration files
  that are easy to store in version control, generate automatically and diff.
  Codebase should be generic and not contain any of the model hyperparameters.
  Unless user intervenes, it should be obvious which model was run
  with which hyperparameters and what output it gave.
- The amount of "magic" in the framework should be limited and it should be easy to
  understand what exactly the model is doing for newcomers already comfortable with PyTorch. 
- All state-of-the-art models should be implemented in the framework with accuracy
  matching published results.
  Currently I'm focusing on computer vision and reinforcement learning models.
- All common deep learning workflows should be fast to implement, while 
  uncommon ones should be possible. At least as far as PyTorch allows.
  
  
# Implemented models - Computer Vision

Several models are already implemented in the framework and have example config files
that are ready to run and easy to modify for other similar usecases:

- State-of-the art results on Cifar10 dataset using residual networks
- Cats vs dogs classification using transfer learning from a resnet34 model pretrained on 
  ImageNet
  
# Implemented models - Reinforcement learning

- Continuous and discrete action spaces
- Advantage Actor-Critic (A2C),
  Proximal Policy Optimization (PPO), 
  Trust Region Policy Optimization (TRPO),
  and 
  Actor-Critic with Experience Replay (ACER)
  policy gradient reinforcement learning algorithms.
- Deep Q-Learning (DQN) as described by DeepMind in their Nature publication with following 
  improvements: Double DQN, Dueling DQN, Prioritized experience replay.


# How to run the examples?

While it is possible to specify your models entirely by code,
framework tries to encourage you to use config files for that purpose.
In the `examples` directory, there are defined multiple working config files showcasing
most popular algorithms with sane default hyperparameters.
 
For example, to run the A2C algorithm on a Breakout atari environment, simply invoke:

```
python -m vel.launcher examples-configs/rl/atari/a2c/breakout_a2c.yaml train
```

If you install the library locally, it will create a script wrapping the launcher for 
you. Then, above becomes:

```
vel examples-configs/rl/atari/a2c/breakout_a2c.yaml train
```

General command line interface of the launcher is:

```
python -m vel.launcher CONFIGFILE COMMAND --device PYTORCH_DEVICE -r RUN_NUMBER -s SEED
```

Where `PYTORCH_DEVICE` is a valid name of pytorch device, most likely `cuda:0`.
Run number is a sequential number you wish to record your results with.

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

# Glossary

For a glossary of terms used in the library please refer to [Glossary](docs/Glossary.md).
If there is anything you'd like to see there, feel free to open an issue or make a pull request.

# Bibliography

For a more or less exhaustive bibliography please refer to [Bibliography](docs/Bibliography.md).

