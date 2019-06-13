import numpy as np
import pytest

import numpy.testing as nt

import torch
import torch.nn.functional as F
import torch.distributions as d

from vel.rl.modules.action_head import DiagGaussianActionHead, CategoricalActionHead


def test_sample_diag_gaussian():
    """ Test sampling from a multivariate gaussian distribution with a diagonal covariance matrix """
    head = DiagGaussianActionHead(1, 5)

    array = np.zeros((10000, 5, 2))

    sample = head.sample(torch.from_numpy(array))

    result_array = sample.detach().cpu().numpy()

    nt.assert_array_less(np.abs(result_array.mean(axis=0)), 0.1)
    nt.assert_array_less(result_array.std(axis=0), 1.1)
    nt.assert_array_less(0.9, result_array.std(axis=0))

    array2 = np.zeros((10000, 5, 2))
    array2[:, 0, 0] = 5.0
    array2[:, 0, 1] = np.log(10)

    sample2 = head.sample(torch.from_numpy(array2))

    result_array2 = sample2.detach().cpu().numpy()
    nt.assert_array_less(result_array2.mean(axis=0), np.array([5.3, 0.1, 0.1, 0.1, 0.1]))
    nt.assert_array_less(np.array([4.7, -0.1, -0.1, -0.1, -0.1]), result_array2.mean(axis=0))


def test_neglogp_diag_gaussian():
    """
    Test negative logarithm of likelihood of a multivariate gaussian distribution with a diagonal covariance matrix
    """
    head = DiagGaussianActionHead(1, 5)
    distrib = d.MultivariateNormal(torch.tensor([1.0, -1.0]), covariance_matrix=torch.tensor([[2.0, 0.0], [0.0, 0.5]]))

    pd_params = torch.tensor([[1.0, -1.0], [np.log(np.sqrt(2.0)), np.log(np.sqrt(0.5))]]).t()
    sample = head.sample(pd_params[None])

    log_prob1 = distrib.log_prob(sample)
    log_prob2 = head.logprob(sample, pd_params[None])

    nt.assert_allclose(log_prob1.detach().cpu().numpy(), log_prob2.detach().cpu().numpy(), rtol=1e-5)


def test_entropy_diag_gaussian():
    """
    Test entropy of a multivariate gaussian distribution with a diagonal covariance matrix
    """
    head = DiagGaussianActionHead(1, 5)
    distrib = d.MultivariateNormal(torch.tensor([1.0, -1.0]), covariance_matrix=torch.tensor([[2.0, 0.0], [0.0, 0.5]]))

    pd_params = torch.tensor([[1.0, -1.0], [np.log(np.sqrt(2.0)), np.log(np.sqrt(0.5))]]).t()

    entropy1 = distrib.entropy()
    entropy2 = head.entropy(pd_params[None])

    nt.assert_allclose(entropy1.detach().cpu().numpy(), entropy2.detach().cpu().numpy())


def test_kl_divergence_diag_gaussian():
    """
    Test kl divergence between multivariate gaussian distributions with a diagonal covariance matrix
    """
    head = DiagGaussianActionHead(1, 5)

    distrib1 = d.MultivariateNormal(torch.tensor([1.0, -1.0]), covariance_matrix=torch.tensor([[2.0, 0.0], [0.0, 0.5]]))
    distrib2 = d.MultivariateNormal(torch.tensor([0.3, 0.7]), covariance_matrix=torch.tensor([[1.8, 0.0], [0.0, 5.5]]))

    pd_params1 = torch.tensor([[1.0, -1.0], [np.log(np.sqrt(2.0)), np.log(np.sqrt(0.5))]]).t()
    pd_params2 = torch.tensor([[0.3, 0.7], [np.log(np.sqrt(1.8)), np.log(np.sqrt(5.5))]]).t()

    kl_div_1 = d.kl_divergence(distrib1, distrib2)
    kl_div_2 = head.kl_divergence(pd_params1[None], pd_params2[None])

    assert kl_div_1.item() == pytest.approx(kl_div_2.item(), 0.001)


def test_sample_categorical():
    """
    Test sampling from a categorical distribution
    """
    head = CategoricalActionHead(1, 5)

    array = np.zeros((10000, 5))

    sample = head.sample(torch.from_numpy(array))

    result_array = sample.detach().cpu().numpy()

    nt.assert_array_less(np.abs(result_array.mean(axis=0)), 2.1)
    nt.assert_array_less(1.9, np.abs(result_array.mean(axis=0)))

    array2 = np.zeros((10000, 5))
    array2[:, 0:4] = -10.0
    array2[:, 4] = 10.0

    sample2 = head.sample(F.log_softmax(torch.from_numpy(array2), dim=1))
    result_array2 = sample2.detach().cpu().numpy()

    nt.assert_array_less(np.abs(result_array2.mean(axis=0)), 4.1)
    nt.assert_array_less(3.9, np.abs(result_array2.mean(axis=0)))


def test_neglogp_categorical():
    """
    Test negative logarithm of likelihood of a categorical distribution
    """
    head = CategoricalActionHead(1, 5)

    logits = F.log_softmax(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), dim=0)

    distrib = d.Categorical(logits=logits)

    actions = torch.tensor([0, 1, 2, 3, 4])

    log_p_1 = distrib.log_prob(actions)
    log_p_2 = head.logprob(actions, torch.stack([logits, logits, logits, logits, logits], dim=0))

    nt.assert_allclose(log_p_1.detach().cpu().numpy(), log_p_2.detach().cpu().numpy(), rtol=1e-5)


def test_entropy_categorical():
    """
    Test entropy of a categorical distribution
    """
    head = CategoricalActionHead(1, 5)

    logits = F.log_softmax(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), dim=0)

    distrib = d.Categorical(logits=logits)

    entropy1 = distrib.entropy()
    entropy2 = head.entropy(logits[None])

    nt.assert_allclose(entropy1.item(), entropy2.item())


def test_kl_divergence_categorical():
    """
    Test KL divergence between categorical distributions
    """
    head = CategoricalActionHead(1, 5)

    logits1 = F.log_softmax(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), dim=0)
    logits2 = F.log_softmax(torch.tensor([-1.0, 0.2, 5.0, 2.0, 8.0]), dim=0)

    distrib1 = d.Categorical(logits=logits1)
    distrib2 = d.Categorical(logits=logits2)

    kl_div_1 = d.kl_divergence(distrib1, distrib2)
    kl_div_2 = head.kl_divergence(logits1[None], logits2[None])

    nt.assert_allclose(kl_div_1.item(), kl_div_2.item(), rtol=1e-5)
