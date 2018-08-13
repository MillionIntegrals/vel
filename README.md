# Vel

This repository is my project to bring **velocity** to deep-learning research,
by providing tried and tested large pool of prebuilt components that are 
known to be working well together.

I would like to minimize time to market of new projects, ease experimentation
and provide bits of experiment management to bring some order to the data science workflow.

Ideally, for most applications it should be enough to write a
config file wiring existing components together.
If that's not the case writing bits of custom code
shouldn't be unnecessarily complex.

This repository is still in an early stage of that journey but it will grow
as I'll be putting some work into it.

# Requirements

This project requires Python 3.7 and PyTorch 0.4.1. Default project configuration writes
metrics to mongodb instance open on localhost port 27017 and visdom instance 
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
  
  
# Implemented models

Several models are already implemented in the framework and have example config files
that are ready to run and easy to modify for other similar usecases:

- State-of-the art results on Cifar10 dataset using residual networks
- Cats vs dogs classification using transfer learning from a resnet34 model pretrained on 
  ImageNet
- Actor-Critic (A2C) and Proximal-Policy-Optimization (PPO) policy gradient reinforcement
  learning algorithms.
- Deep Q-Learning (DQN) as described by DeepMind in their research paper with following 
  improvements: Double DQN, Dueling DQN, Prioritized experience replay.


# How to run the examples?

Whole framework is built around the idea of running config files from code. For example,
to run the Actor-Critic algorithm simply invoke:

```
python -m vel.launcher examples/rl/atari/a2c/breakout_a2c.yaml train
```

General form of the call is as follows 


```
python -m vel.launcher CONFIGFILE COMMAND --device PYTORCH_DEVICE -r RUN_NUMBER
```

Where `PYTORCH_DEVICE` is a valid name of pytorch device, most probably `cuda:0`, and run
number is the sequential number of run you wish to record your results under.

# Glossary

For a glossary of terms used in the library please refer to [Glossary](docs/Glossary.md).
If there is anything you'd like to see there, feel free to open an issue or make a pull request.

# Bibliography

For a more or less exhaustive bibliography please refer to [Bibliography](docs/Bibliography.md).

