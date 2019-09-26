import torch.nn.utils

from vel.model.latent.vae_base import VaeBase


class IWAE(VaeBase):
    """
    Importance-Weighted Auto-Encoder https://arxiv.org/abs/1509.00519
    """

    def __init__(self, k: int = 5, analytical_kl_div=True):
        super().__init__(analytical_kl_div=analytical_kl_div)

        self.k = k

    def calculate_gradient(self, data: dict) -> dict:
        """ Calculate model gradient for given data sample """
        encoded = self.encoder_network(data['x'])
        z_dist = self.encoder_distribution(encoded)

        bs = encoded.size(0)
        # Encode importance samples into batch dimension for the decoded network
        z = z_dist.rsample([self.k]).reshape([bs * self.k, -1])

        decoded = self.decoder_network(z)
        decoded = decoded.reshape([self.k, bs] + list(decoded.shape[1:]))

        # Unpack to make distribution efficient for broadcasting
        x_dist = self.decoder_distribution(decoded)
        prior = self.prior_distribution()

        kl_divergence = self.kl_divergence(z, z_dist, prior)
        reconstruction = x_dist.log_prob(data['y'])

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

        # Perform log-mean-exp on the axis of importance samples
        # Then mean across batch
        elbo = self.log_mean_exp(elbo, 0).mean()

        loss = -elbo

        if self.training:
            loss.backward()

        with torch.no_grad():
            return {
                'loss': loss.item(),
                'reconstruction': -reconstruction.mean().item(),
                'kl_divergence': kl_divergence.mean().item()
            }
