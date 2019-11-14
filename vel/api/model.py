import torch

import vel.util.module_util as mu

from vel.api.optimizer import VelOptimizer, OptimizerFactory
from vel.metric.loss_metric import Loss


from .vmodule import VModule


class Model(VModule):
    """ Class representing full neural network model, generally used to solve some problem """

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return []

    def train(self, mode=True):
        """
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super().train(mode)

        if mode:
            mu.apply_leaf(self, mu.set_train_mode)

        return self

    def summary(self):
        """ Print a model summary """
        print(self)
        print("-" * 100)
        for name, module in self.named_parameters():
            print("> {} {:,}".format(name, module.numel()))
        print("-" * 100)
        number = sum(p.numel() for p in self.parameters())
        print("Number of model parameters: {:,}".format(number))
        print("-" * 100)


class OptimizedModel(Model):
    """ Model that is being optimized by an 'optimizer' """

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        return optimizer_factory.instantiate(parameters)

    def optimize(self, data: dict, optimizer: VelOptimizer) -> dict:
        """
        Perform one step of optimization of the model
        :returns a dictionary of metrics
        """
        raise NotImplementedError


class ValidatedModel(OptimizedModel):
    """ Model that also has a validation operation """

    def validate(self, data: dict) -> dict:
        """
        Perform one step of model inference without optimization
        :returns a dictionary of metrics
        """
        raise NotImplementedError


class GradientModel(ValidatedModel):
    """ Model that calculates a single gradient and optimizes it """

    def optimize(self, data: dict, optimizer: VelOptimizer) -> dict:
        """
        Perform one step of optimization of the model
        :returns a dictionary of metrics
        """
        optimizer.zero_grad()

        metrics = self.calculate_gradient(data)

        opt_metrics = optimizer.step()

        for key, value in opt_metrics.items():
            metrics[key] = value

        return metrics

    @torch.no_grad()
    def validate(self, data: dict) -> dict:
        """
        Perform one step of model inference without optimization
        :returns a dictionary of metrics
        """
        return self.calculate_gradient(data)

    def calculate_gradient(self, data: dict) -> dict:
        """
        Calculate gradient for given batch of training data.
        :returns a dictionary of metrics
        """
        raise NotImplementedError


class LossFunctionModel(GradientModel):
    """ Model for a supervised learning with a simple loss function """

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return [Loss()]

    def calculate_gradient(self, data: dict) -> dict:
        if self.is_stateful:
            y_hat, _ = self(data['x'])
        else:
            y_hat = self(data['x'])

        loss_value = self.loss_value(data['x'], data['y'], y_hat)

        if self.training:
            loss_value.backward()

        return {
            'loss': loss_value.item(),
            'data': data['x'],
            'target': data['y'],
            'output': y_hat
        }

    def loss_value(self, x_data, y_true, y_pred) -> torch.tensor:
        """ Calculate a value of loss function """
        raise NotImplementedError
