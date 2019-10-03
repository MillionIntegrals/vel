"""
VQ-VAE implementation with Vector Quantization functions taken from
https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""
import itertools as it

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import vel.util.network as net_util

from vel.api import GradientModel
from vel.metric import AveragingNamedMetric
from vel.metric.loss_metric import Loss


class VectorQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            'Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.'
        )


class VectorQuantizationStraightThrough(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vector_quantization(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()

        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous().view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return grad_inputs, grad_codebook


vector_quantization = VectorQuantization.apply
vector_quantization_straight_through = VectorQuantizationStraightThrough.apply


class VQEmbedding(nn.Module):
    """ Vector-Quantised code embedding for the latent variables """

    def __init__(self, k: int, d: int):
        super().__init__()
        self.k = k
        self.d = d
        self.embedding = nn.Parameter(torch.empty((self.k, self.d)))

    def reset_weights(self):
        """ Initialize weights of the embedding """
        self.embedding.data.uniform_(-1.0/self.k, 1.0/self.k)

    def extra_repr(self) -> str:
        return f"k={self.k}, d={self.d}"

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vector_quantization(z_e_x_, self.embedding)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vector_quantization_straight_through(z_e_x_, self.embedding.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class VQVAE(GradientModel):
    """
    Implementation of Neural Discrete Representation Learning  (https://arxiv.org/abs/1711.00937)
    Vector-Quantised Variational-AutoEncoder (VQ-VAE)
    """

    def __init__(self, img_rows, img_cols, img_channels, channels=None, k: int = 512, d: int = 256,
                 beta: float = 1.0):
        super().__init__()

        if channels is None:
            channels = [16, 32, 32]

        layer_series = [
            (3, 1, 1),
            (3, 1, 2),
            (3, 1, 2),
        ]

        self.codebook = VQEmbedding(k, d)

        self.final_width = net_util.convolutional_layer_series(img_rows, layer_series)
        self.final_height = net_util.convolutional_layer_series(img_cols, layer_series)
        self.channels = channels

        self.beta = beta
        self.k = k
        self.d = d

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=channels[0], kernel_size=(3, 3), padding=1),
            nn.SELU(True),
            nn.LayerNorm([
                channels[0],
                net_util.convolutional_layer_series(img_rows, layer_series[:1]),
                net_util.convolutional_layer_series(img_cols, layer_series[:1]),
            ]),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3), stride=2, padding=1),
            nn.SELU(True),
            nn.LayerNorm([
                channels[1],
                net_util.convolutional_layer_series(img_rows, layer_series[:2]),
                net_util.convolutional_layer_series(img_cols, layer_series[:2]),
            ]),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3), stride=2, padding=1),
            nn.SELU(True),
            nn.LayerNorm([
                channels[2],
                net_util.convolutional_layer_series(img_rows, layer_series),
                net_util.convolutional_layer_series(img_cols, layer_series),
            ]),
            nn.Conv2d(in_channels=channels[2], out_channels=self.d, kernel_size=(3, 3), stride=1, padding=1),
            nn.SELU(True),
            nn.LayerNorm([
                self.d,
                net_util.convolutional_layer_series(img_rows, layer_series),
                net_util.convolutional_layer_series(img_cols, layer_series),
            ]),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.d, out_channels=channels[2], kernel_size=(3, 3), stride=1, padding=1),
            # nn.Linear(d, self.final_width * self.final_height * channels[2]),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[2],
                net_util.convolutional_layer_series(img_rows, layer_series),
                net_util.convolutional_layer_series(img_cols, layer_series),
            ]),
            nn.ConvTranspose2d(
                in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[1],
                net_util.convolutional_layer_series(img_rows, layer_series[:2]),
                net_util.convolutional_layer_series(img_cols, layer_series[:2]),
            ]),
            nn.ConvTranspose2d(
                in_channels=channels[1], out_channels=channels[0], kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            # nn.ReLU(True),
            nn.SELU(True),
            nn.LayerNorm([
                channels[0],
                net_util.convolutional_layer_series(img_rows, layer_series[:1]),
                net_util.convolutional_layer_series(img_cols, layer_series[:1]),
            ]),
            nn.ConvTranspose2d(in_channels=channels[0], out_channels=img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reset_weights(self):
        self.codebook.reset_weights()

        for m in it.chain(self.encoder, self.decoder):
            if isinstance(m, nn.Conv2d):
                self._weight_initializer(m)
            elif isinstance(m, nn.ConvTranspose2d):
                self._weight_initializer(m)
            elif isinstance(m, nn.Linear):
                self._weight_initializer(m)

    @staticmethod
    def _weight_initializer(tensor):
        init.xavier_uniform_(tensor.weight, gain=init.calculate_gain('relu'))
        init.constant_(tensor.bias, 0.0)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde

    def calculate_gradient(self, data: dict) -> dict:
        """
        Calculate gradient for given batch of supervised learning.
        Returns a dictionary of metrics
        """
        input_data = data['x']
        target_data = data['y']

        # x_tilde, z_e_x, z_q_x = self(input_data)
        z_e_x = self.encoder(input_data)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, target_data)

        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + self.beta * loss_commit

        if self.training:
            loss.backward()

        return {
            'loss': loss.item(),

            'grad_norm': grad_norm,
            'reconstruction': loss_recons.item(),
            'loss_vq': loss_vq.item(),
            'loss_commit': loss_commit.item()
        }

    def metrics(self):
        """ Set of metrics for this model """
        return [
            Loss(),
            AveragingNamedMetric('reconstruction', scope="train"),
            AveragingNamedMetric('loss_vq', scope="train"),
            AveragingNamedMetric('loss_commit', scope="train"),
            AveragingNamedMetric('grad_norm', scope="train")
        ]


def create(img_rows, img_cols, img_channels, channels=None, k: int = 512, d: int = 256,
           beta: float = 1.0):
    """ Vel factory function """
    from vel.api import ModelFactory

    if channels is None:
        channels = [16, 32, 32]

    def instantiate(**_):
        return VQVAE(
            img_rows, img_cols, img_channels, channels=channels, k=k, d=d, beta=beta
        )

    return ModelFactory.generic(instantiate)
