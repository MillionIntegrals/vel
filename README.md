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
  Unless user intervenes, it should be obvious which model was run
  with which hyperparameters and what output it gave.
- (Ongoing) The amount of "magic" in the framework should be limited and it should be easy to
  understand what exactly the model is doing for newcomers already comfortable with PyTorch. 
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
- [x] Learning rate finder
- [ ] Cats vs dogs transfer learning
- [ ] "Phase" training
- [ ] Notebook integration
- [ ] SGD with warm restarts
- [ ] Super-convergence experiments


## Smaller TODO items:

- [ ] Write output of LrFinder to storage rather than to matplotlib
- [ ] Fork torch summary and incorporate into the framework


## Important links:

- https://pytorch.org/tutorials/
- https://pytorch.org/docs/stable/index.html
- https://github.com/pytorch/vision
- https://github.com/pytorch/tnt
- https://github.com/fastai/fastai