# waterboy
My goal is to establish a Rails-like convention-over-configuration opinionated set of
tools streamlining research and development of deep learning models.

I wanted to minimize time to market of new projects, ease experimentation
and combine that with experiment management to bring some order to an already 
noisy data science workflow.

# Features

- (Done) Models should be runnable from the configuration files
  that are easy to store in version control, generate automatically and diff.
  Codebase should be generic and not contain any of the model hyperparameters.
  Unless user intervenes, it should be obvious which model was run when with which 
  hyperparameters gave which outputs.
- (Ongoing) All state-of-the-art models should be implemented in the framework with accuracy
  matching published results.
  Currently I'm focusing on computer vision and reinforcement learning models.
- (Done) All common deep learning workflows should be fast to implement, while 
  uncommon ones should be possible. At least as far as PyTorch allows.



## TODO:

- [x] Metric/aggregation API
- [x] Training progress pretty print
- [x] Integrate loss/metrics with a model
- [x] CIFAR 10 dataset and models
- [x] CIFAR 10 Resnet models
- [x] Refine source API
- [x] Augmentation visualization
- [x] Learning rate scheduling
- [x] Augmentations and input transformation pipeline
- [x] CIFAR 10 state of the art
- [x] Database storage of experiment results
- [x] Model checkpointing
- [x] Visdom integration
- [ ] Notebook integration
- [ ] Learning rate finder
- [ ] SGD with warm restarts
- [ ] "Phase" training
- [ ] Super-convergence experiments
- [ ] Cats vs dogs transfer learning


## Important links:

- https://pytorch.org/tutorials/
- https://pytorch.org/docs/stable/index.html
- https://github.com/pytorch/vision
- https://github.com/pytorch/tnt