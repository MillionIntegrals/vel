import torchvision.models.resnet as m
import torch.nn as nn
import torch.nn.functional as F

import vel.module.layers as layers
import vel.util.module_util as mu

from vel.api import LossFunctionModel, ModuleFactory, OptimizerFactory, VelOptimizer


# Because of concat pooling it's 2x 512
NET_OUTPUT = 1024


class Resnet34(LossFunctionModel):
    """ Resnet34 network model """

    def __init__(self, fc_layers=None, dropout=None, pretrained=True):
        super().__init__()

        # Store settings, maybe someone will be interested to see them
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.pretrained = pretrained

        self.head_layers = 8
        self.group_cut_layers = (6, 10)

        # Load backbbone
        backbone = m.resnet34(pretrained=pretrained)

        # If fc layers is set, let's put custom head
        if fc_layers:
            # Take out the old head and let's put the new head
            valid_children = list(backbone.children())[:-2]

            valid_children.extend([
                layers.AdaptiveConcatPool2d(),
                layers.Flatten()
            ])

            layer_inputs = [NET_OUTPUT] + fc_layers[:-1]

            dropout = dropout or [None] * len(fc_layers)

            for idx, (layer_input, layet_output, layer_dropout) in enumerate(zip(layer_inputs, fc_layers, dropout)):
                valid_children.append(nn.BatchNorm1d(layer_input))

                if layer_dropout:
                    valid_children.append(nn.Dropout(layer_dropout))

                valid_children.append(nn.Linear(layer_input, layet_output))

                if idx == len(fc_layers) - 1:
                    # Last layer
                    valid_children.append(nn.LogSoftmax(dim=1))
                else:
                    valid_children.append(nn.ReLU())

            final_model = nn.Sequential(*valid_children)
        else:
            final_model = backbone

        self.model = final_model

    def freeze(self, groups=None):
        """ Freeze given number of layers in the model """
        layer_groups = dict(self.layer_groups())

        if groups is None:
            groups = layer_groups.keys()

        for group in groups:
            for module in layer_groups[group]:
                mu.freeze_layer(module)

    def unfreeze(self):
        """ Unfreeze model layers """
        for idx, child in enumerate(self.model.children()):
            mu.unfreeze_layer(child)

    def layer_groups(self):
        """ Return layers grouped """
        g1 = list(self.model[:self.group_cut_layers[0]])
        g2 = list(self.model[self.group_cut_layers[0]:self.group_cut_layers[1]])
        g3 = list(self.model[self.group_cut_layers[1]:])

        return [
            ('top', g1),
            ('mid', g2),
            ('bottom', g3)
        ]

    def parameter_groups(self):
        return [(name, mu.module_list_to_param_list(m)) for name, m in self.layer_groups()]

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        return optimizer_factory.instantiate(self.parameter_groups())

    def forward(self, x):
        """ Calculate model value """
        return self.model(x)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate value of the loss function """
        return F.nll_loss(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        from vel.metric.loss_metric import Loss
        from vel.metric.accuracy import Accuracy
        return [Loss(), Accuracy()]


def create(fc_layers=None, dropout=None, pretrained=True):
    """ Vel factory function """
    def instantiate(**_):
        return Resnet34(fc_layers, dropout, pretrained)

    return ModuleFactory.generic(instantiate)
