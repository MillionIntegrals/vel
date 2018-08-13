"""
Code based on
https://github.com/fastai/fastai/blob/master/fastai/model.py
"""
import torch.nn as nn

import itertools as it


def is_listy(x):
    return isinstance(x, (list, tuple))


def model_children(module):
    return module if isinstance(module, (list, tuple)) else list(module.children())


def apply_leaf(module, f):
    c = model_children(module)

    if isinstance(module, nn.Module):
        f(module)

    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_train_mode(module):
    # Only fix ones which we don't want to "train"
    if hasattr(module, 'running_mean') and (getattr(module, 'bn_freeze', False) or not getattr(module, 'trainable', True)):
        module.eval()
    elif getattr(module, 'drop_freeze', False) and hasattr(module, 'p') and ('drop' in type(module).__name__.lower()):
        module.eval()


def set_trainable_attr(module, trainable):
    module.trainable = trainable


def set_requires_gradient(module, trainable):
    for p in module.parameters():
        p.requires_grad = trainable


def freeze_layer(module):
    apply_leaf(module, lambda x: set_trainable_attr(x, trainable=False))
    set_requires_gradient(module, trainable=False)


def unfreeze_layer(module):
    apply_leaf(module, lambda x: set_trainable_attr(x, trainable=True))
    set_requires_gradient(module, trainable=True)


def trainable_params_(m):
    """ Returns a list of trainable parameters in the model m. (i.e., those that require gradients.) """
    return [p for p in m.parameters() if p.requires_grad]


def chain_params(p):
    if is_listy(p):
        return list(it.chain(*[trainable_params_(o) for o in p]))
    return trainable_params_(p)


def to_parameter_groups(layer_groups):
    """ Convert from list of layer groups into list of parameter settings for an optimizer """
    return [{'params': chain_params(x)} for x in layer_groups]
