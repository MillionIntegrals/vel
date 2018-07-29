# waterboy

My goal is to establish a Rails-like convention-over-configuration opinionated set of
tools streamlining research and development of deep learning models.

I wanted to minimize time to market of new projects, ease experimentation
and combine that with experiment management to bring some order to an already 
noisy data science workflow.

This repository is still in an early stage of that journey but it will grow
as I'll be putting some work into that.


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
  
  
# Examples

Several models are already implemented in the framework and have prepared config files
that are ready to run and easy to modify for other similar usecases:

- State-of-the art results on Cifar10 dataset using residual networks
- Cats vs dogs classification using transfer learning from a resnet34 model pretrained on 
  ImageNet
- Actor-Critic policy gradient reinforcement algorithm


## Important links:

- https://pytorch.org/tutorials/
- https://pytorch.org/docs/stable/index.html
- https://github.com/pytorch/vision
- https://github.com/fastai/fastai
- https://github.com/openai/baselines