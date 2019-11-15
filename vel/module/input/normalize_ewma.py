import torch

from vel.api import VModule


class NormalizeEwma(VModule):
    """ Normalize a vector of observations - across the batch dim """

    def __init__(self, input_shape, beta=0.99, per_element_update=False, epsilon=1e-1):
        super().__init__()

        self.input_shape = input_shape
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.register_buffer('running_mean', torch.zeros(input_shape, dtype=torch.float))
        self.register_buffer('running_var', torch.ones(input_shape, dtype=torch.float))
        self.register_buffer('debiasing_term', torch.tensor(self.epsilon, dtype=torch.float))

    def reset_weights(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.debiasing_term.fill_(self.epsilon)

    def forward(self, input_vector):
        # Make sure input is float32
        input_vector = input_vector.to(torch.float)

        if self.training:
            batch_mean = input_vector.mean(dim=0)
            batch_var = input_vector.var(dim=0, unbiased=False)

            if self.per_element_update:
                batch_size = input_vector.size(0)
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_var.mul_(weight).add_(batch_var * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        debiased_mean = self.running_mean / self.debiasing_term
        debiased_var = self.running_var / self.debiasing_term

        return (input_vector - debiased_mean.unsqueeze(0)) / torch.sqrt(debiased_var.unsqueeze(0))
