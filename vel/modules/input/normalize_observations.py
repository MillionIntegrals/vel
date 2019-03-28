import torch
import numbers

from vel.api import BackboneModel, ModelFactory


class NormalizeObservations(BackboneModel):
    """ Normalize a vector of observations """

    def __init__(self, input_shape, epsilon=1e-6):
        super().__init__()

        self.input_shape = input_shape
        self.epsilon = epsilon

        self.register_buffer('running_mean', torch.zeros(input_shape, dtype=torch.float))
        self.register_buffer('running_var', torch.ones(input_shape, dtype=torch.float))
        self.register_buffer('count', torch.tensor(epsilon, dtype=torch.float))

    def reset_weights(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.count.fill_(self.epsilon)

    def forward(self, input_vector):
        # Make sure input is float32
        input_vector = input_vector.to(torch.float)

        if self.training:
            batch_mean = input_vector.mean(dim=0)
            batch_var = input_vector.var(dim=0, unbiased=False)
            batch_count = input_vector.size(0)

            delta = batch_mean - self.running_mean
            tot_count = self.count + batch_count

            self.running_mean.add_(delta * batch_count / tot_count)

            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            self.count.add_(batch_count)
            self.running_var.copy_(new_var)

        return (input_vector - self.running_mean.unsqueeze(0)) / torch.sqrt(self.running_var.unsqueeze(0))


def create(input_shape):
    """ Vel factory function """
    if isinstance(input_shape, numbers.Number):
        input_shape = (input_shape,)
    elif not isinstance(input_shape, tuple):
        input_shape = tuple(input_shape)

    def instantiate(**_):
        return NormalizeObservations(input_shape)

    return ModelFactory.generic(instantiate)


# Scripting interface
NormalizeObservationsFactory = create
