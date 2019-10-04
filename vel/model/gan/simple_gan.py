"""
Simple GAN code is based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""
import collections
import numpy as np
import torch
import torch.nn as nn

from vel.api import OptimizedModel, ModelFactory, VelOptimizer, OptimizerFactory
from vel.api.optimizer import VelMultiOptimizer
from vel.metric import AveragingNamedMetric


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class SimpleGAN(OptimizedModel):
    """
    Implements simple Generative Adversarial Network in the spirit of the original paper
    "Generative Adversarial Networks" https://arxiv.org/abs/1406.2661
    """

    def __init__(self, img_rows, img_cols, img_channels, latent_dim):
        super().__init__()

        self.image_shape = (img_channels, img_rows, img_cols)
        self.latent_dim = latent_dim

        self.generator = Generator(img_shape=self.image_shape, latent_dim=self.latent_dim)
        self.discriminator = Discriminator(img_shape=self.image_shape, latent_dim=self.latent_dim)

        self.adversarial_loss = nn.BCELoss()

    def create_optimizer(self, optimizer_factory: OptimizerFactory) -> VelMultiOptimizer:
        """ Create optimizer for the purpose of optimizing this model """
        gen_parameters = filter(lambda p: p.requires_grad, self.generator.parameters())
        disc_parameters = filter(lambda p: p.requires_grad, self.discriminator.parameters())

        return optimizer_factory.instantiate_multi(collections.OrderedDict([
            ('generator', gen_parameters),
            ('discriminator', disc_parameters)
        ]))

    def optimize(self, data: dict, optimizer: VelMultiOptimizer) -> dict:
        """
        Perform one step of optimization of the model
        :returns a dictionary of metrics
        """
        optimizer_G = optimizer['generator']
        optimizer_D = optimizer['discriminator']

        input_data = data['x']

        # Adversarial ground truths
        valid = torch.ones(input_data.size(0), 1).to(input_data.device)
        fake = torch.zeros(input_data.size(0), 1).to(input_data.device)

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(input_data.size(0), self.latent_dim).to(input_data.device)

        # Generate a batch of images
        gen_imgs = self.generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(input_data), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        return {
            'gen_loss': g_loss.item(),
            'disc_loss': d_loss.item()
        }

    def validate(self, data: dict) -> dict:
        """
        Perform one step of model inference without optimization
        :returns a dictionary of metrics
        """
        return {
            'gen_loss': 0.0,
            'disc_loss': 0.0
        }

    def metrics(self):
        """ Set of metrics for this model """
        return [
            AveragingNamedMetric('gen_loss', scope="train"),
            AveragingNamedMetric('disc_loss', scope="train"),
        ]


def create(img_rows, img_cols, img_channels, latent_dim):
    """ Vel factory function """
    def instantiate(**_):
        return SimpleGAN(
            img_rows, img_cols, img_channels, latent_dim=latent_dim
        )

    return ModelFactory.generic(instantiate)
