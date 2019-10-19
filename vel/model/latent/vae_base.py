import torch
import torch.distributions as dist

from vel.api import GradientModel
from vel.metric import AveragingNamedMetric
from vel.metric.loss_metric import Loss


class VaeBase(GradientModel):
    """ Base module for variational auto-encoder implementations """

    def __init__(self, analytical_kl_div=True):
        super().__init__()

        self.analytical_kl_div = analytical_kl_div

    ####################################################################################################################
    # Interface methods
    def encoder_network(self, sample: torch.Tensor) -> torch.Tensor:
        """ Transform input sample into an encoded representation """
        raise NotImplementedError

    def encoder_distribution(self, encoded: torch.Tensor) -> dist.Distribution:
        """ Create a pytorch distribution object representing the encoder distribution (approximate posterior) """
        raise NotImplementedError

    def decoder_network(self, z: torch.Tensor) -> torch.Tensor:
        """ Transform encoded value into a decoded representation """
        raise NotImplementedError

    def decoder_distribution(self, decoded: torch.Tensor) -> dist.Distribution:
        """ Create a pytorch distribution object representing the decoder distribution (likelihood) """
        raise NotImplementedError

    def prior_distribution(self) -> dist.Distribution:
        """ Return a prior distribution object """
        raise NotImplementedError

    ####################################################################################################################
    # Other useful methods
    def encoder_rsample(self, encoded: torch.Tensor) -> torch.Tensor:
        """ Sample with "reparametrization trick" encoder sample """
        return self.encoder_distribution(encoded).rsample()

    def decoder_sample(self, decoded: torch.Tensor) -> torch.Tensor:
        """ Sample from a decoder distribution """
        return self.decoder_distribution(decoded).sample()

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        """ Encode incoming data into a latent representation """
        encoded = self.encoder_network(sample)
        return self.encoder_rsample(encoded)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back into data domain.
        Sample from p(x | z)
        """
        decoded = self.decoder_network(z)
        return self.decoder_sample(decoded)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """ Simple forward pass through the module """
        z = self.encode(sample)
        decoded = self.decode(z)
        return decoded

    def calculate_gradient(self, data: dict) -> dict:
        """ Calculate model gradient for given data sample """
        encoded = self.encoder_network(data['x'])
        z_dist = self.encoder_distribution(encoded)
        z = z_dist.rsample()

        decoded = self.decoder_network(z)
        x_dist = self.decoder_distribution(decoded)
        prior = self.prior_distribution()

        kl_divergence = self.kl_divergence(z, z_dist, prior).mean()
        reconstruction = x_dist.log_prob(data['y']).mean()

        # ELBO is E_q log p(x, z) / q(z | x)
        # Which can be expressed in many equivalent forms:
        # (1) E_q log p(x | z) + log p(z) - log q(z | x)
        # (2) E_q log p(x | z) - D_KL(p(z) || q(z | x))
        # (3) E_q log p(x) - D_KL(p(z | x) || q(z | x)Biblio)

        # Form 3 is interesting from a theoretical standpoint, but is intractable to compute directly
        # While forms (1) and (2) can be computed directly.
        # Positive aspect of form (2) is that KL divergence can be calculated analytically
        # further reducing the variance of the gradient
        elbo = reconstruction - kl_divergence

        loss = -elbo

        if self.training:
            loss.backward()

        return {
            'loss': loss.item(),
            'reconstruction': -reconstruction.item(),
            'kl_divergence': kl_divergence.item()
        }

    def kl_divergence(self, z, encoder_distribution, prior) -> torch.Tensor:
        """ Calculate the kl divergence between q(z|x) and p(z) """
        if self.analytical_kl_div:
            kl_divergence = dist.kl_divergence(encoder_distribution, prior)
        else:
            lpz = prior.log_prob(z)
            lqzx = encoder_distribution.log_prob(z)
            kl_divergence = -lpz + lqzx

        return kl_divergence

    def metrics(self):
        """ Set of metrics for this model """
        return [
            Loss(),
            AveragingNamedMetric('reconstruction', scope="train"),
            AveragingNamedMetric('kl_divergence', scope="train"),
        ]

    @torch.no_grad()
    def nll(self, sample: torch.Tensor, num_posterior_samples: int = 1):
        """
        Upper bound on negative log-likelihood of supplied data.
        If num samples goes to infinity, the nll of data should
        approach true value
        """
        assert num_posterior_samples >= 1, "Need at least one posterior sample"

        encoded = self.encoder_network(sample)
        z_dist = self.encoder_distribution(encoded)
        prior = self.prior_distribution()

        if self.analytical_kl_div:
            kl_divergence = dist.kl_divergence(z_dist, prior)

        bs = encoded.size(0)
        z = z_dist.rsample((num_posterior_samples,))

        # Reshape, decode, reshape
        z_reshaped = z.view([bs * num_posterior_samples] + list(z.shape[2:]))
        decoded = self.decoder_network(z_reshaped)
        decoded = decoded.view([num_posterior_samples, bs] + list(decoded.shape[1:]))

        x_dist = self.decoder_distribution(decoded)

        if not self.analytical_kl_div:
            lpz = prior.log_prob(z)
            lqzx = z_dist.log_prob(z)
            kl_divergence = -lpz + lqzx

        likelihood = x_dist.log_prob(sample)
        elbo = likelihood - kl_divergence

        return -self.log_mean_exp(elbo, dim=0)

    ####################################################################################################################
    # Utility methods
    def log_mean_exp(self, inputs, dim=1):
        """ Perform log(mean(exp(data))) in a numerically stable way """
        if inputs.size(dim) == 1:
            return inputs
        else:
            input_max = inputs.max(dim, keepdim=True)[0]
            return (inputs - input_max).exp().mean(dim).log() + input_max.squeeze(dim=dim)
