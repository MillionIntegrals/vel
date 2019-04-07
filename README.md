# Vel 0.3

[![Build Status](https://travis-ci.org/MillionIntegrals/vel.svg?branch=master)](https://travis-ci.org/MillionIntegrals/vel)
[![PyPI version](https://badge.fury.io/py/vel.svg)](https://badge.fury.io/py/vel)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/MillionIntegrals/vel/blob/master/LICENSE)
[![Gitter chat](https://badges.gitter.im/MillionIngegrals/vel.png)](https://gitter.im/deep-learning-vel)


Bring **velocity** to deep-learning research.


This project hosts a collection of **highly modular** deep learning components that are tested to be working well together.
A simple yaml-based system ties these modules together declaratively using configuration files,
but everything that can be defined using config files can be coded directly in the python script as well.


This is still an early version and a hobby project so documentation is unfortunately nonexistent. I've tried to make the
code as clear as possible, and provide many usage examples, but whenever there was a tradeoff to be made between 
simplicity and modularity I've chosen modularity first and simplicity second.


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


# Blogposts

- https://blog.millionintegrals.com/vel-pytorch-meets-baselines/


# How to run it 

Project can be installed from PyPi via `pip install vel` but also can be checked out from github
and installed directly by running
```bash
pip install -e .
```
from the repository root directory.

This project requires Python at least 3.6 and PyTorch 1.0.
If you want to run YAML config examples, you'll also need a **project configuration file**
`.velproject.yaml`. An example is included in this repository.

Default project configuration writes
metrics to MongoDB instance open on localhost port 27017 and Visdom instance 
on localhost port 8097. 

If you don't want to run these services, there is included
another example file `.velproject.dummy.yaml`
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
- All common deep learning workflows should be fast to implement, while 
  uncommon ones should be possible, at least as far as PyTorch allows.
  
  
# Implemented models - Computer Vision

Several models are already implemented in the framework and have example config files
that are ready to run and easy to modify for other similar usecases:

- State-of-the art results on Cifar10 dataset using residual networks
- Cats vs dogs classification using transfer learning from a resnet34 model pretrained on 
  ImageNet
  
# Implemented models - Natural language processing

- Character-level language models based on LSTM and GRU recurrent networks, with example trained on
  works of Shakespeare
- Sentiment analysis of IMDB movie reviews
  
# Implemented models - Reinforcement learning

- Continuous and discrete action spaces
- Basic support for LSTM policies for A2C and PPO
- Following published policy gradient reinforcement learning algorithms:
    - Advantage Actor-Critic (A2C)
    - Deep Deterministic Policy Gradient (DDPG)
    - Proximal Policy Optimization (PPO)
    - Trust Region Policy Optimization (TRPO)
    - Actor-Critic with Experience Replay (ACER)
- Deep Q-Learning (DQN) as described by DeepMind in their Nature publication with following 
  improvements:
    - Double DQN
    - Dueling DQN
    - Prioritized experience replay
    - N-Step Bellman updates
    - Distributional Q-Learning
    - Noisy Networks for Exploration
    - Rainbow (combination of the above)


# Examples

Most of the examples for this framework are defined using config files in the
`examples-configs` directory with sane default hyperparameters already selected.

For example, to run the A2C algorithm on a Breakout atari environment, simply invoke:

```
python -m vel.launcher examples-configs/rl/atari/a2c/breakout_a2c.yaml train
```

If you install the library locally, you'll have a special wrapper created
that will invoke the launcher for you. Then, above becomes:

```
vel examples-configs/rl/atari/a2c/breakout_a2c.yaml train
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

Here is an example script running the same setup as a config file from above:

```python
import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper

from vel.modules.input.image_to_tensor import ImageToTensorFactory
from vel.rl.models.stochastic_policy_model import StochasticPolicyModelFactory
from vel.rl.models.backbone.nature_cnn import NatureCnnFactory


from vel.rl.reinforcers.on_policy_iteration_reinforcer import (
    OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
)

from vel.rl.algo.policy_gradient.a2c import A2CPolicyGradient
from vel.rl.env_roller.step_env_roller import StepEnvRoller

from vel.api.info import TrainingInfo, EpochInfo


def breakout_a2c():
    device = torch.device('cuda:0')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    # Create 16 environments evaluated in parallel in sub processess with all usual DeepMind wrappers
    # These are just helper functions for that
    vec_env = SubprocVecEnvWrapper(
        ClassicAtariEnv('BreakoutNoFrameskip-v4'), frame_history=4
    ).instantiate(parallel_envs=16, seed=seed)

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = StochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnFactory(input_width=84, input_height=84, input_channels=4)
    ).instantiate(action_space=vec_env.action_space)

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            batch_size=256,
            number_of_steps=5,
        ),
        model=model,
        algo=A2CPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.5,
            discount_factor=0.99,
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
        )
    )

    # Model optimizer
    optimizer = optim.RMSprop(reinforcer.model.parameters(), lr=7.0e-4, eps=1e-3)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[StdoutStreaming()]  # Print live metrics every epoch to standard output
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 100 batches per epoch to average metrics nicely
    num_epochs = int(1.1e7 / (5 * 16) / 100)

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=100,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)

    training_info.on_train_end()


if __name__ == '__main__':
    breakout_a2c()
```

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
- Popart reward normalization
- Parameter Space Noise for Exploration
- Hindsight experience replay
- Generative adversarial networks


Code quality:
- Rename models to policies
- Force dictionary inputs and outputs for policies
- Factor action noise back into the policy
- Use linter as a part of the build process


# Alternatives, similar projects

- https://github.com/NervanaSystems/coach
- https://github.com/google/dopamine
- https://github.com/openai/baselines
- https://github.com/rlpy/rlpy
- https://github.com/rlworkgroup/garage
- https://github.com/unixpickle/anyrl-py
- https://github.com/zuoxingdong/lagom
- https://github.com/inoryy/reaver-pysc2
