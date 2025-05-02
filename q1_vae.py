"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""

import math

import numpy as np
import torch

torch.manual_seed(42)


def log_likelihood_bernoulli(mu, target):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # TODO: compute log_likelihood_bernoulli
    ll_bernoulli = target * torch.log(mu) + (1.0 - target) * torch.log(1.0 - mu)
    return ll_bernoulli.sum(-1)


def log_likelihood_normal(mu, logvar, z):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # TODO: compute log normal
    log_of_2pi = torch.log(torch.tensor(2 * torch.pi))
    ll_normal = -0.5 * (log_of_2pi + logvar + (z - mu) ** 2 / logvar.exp())
    ll_normal = ll_normal.sum(-1)

    return ll_normal


def log_mean_exp(y):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # TODO: compute log_mean_exp

    # first normalize the y function by computing its max over sample size and normalizing down
    max_y = torch.max(y, dim=-1, keepdim=True).values
    lme = max_y + torch.log(torch.mean(torch.exp(y - max_y), dim=-1))
    lme = lme.squeeze()

    return lme


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # TODO: compute kld
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl_per_dim = logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0

    kl_gg = 0.5 * torch.sum(kl_per_dim, dim=-1)
    return kl_gg


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = (
        mu_q.view(batch_size, -1)
        .unsqueeze(1)
        .expand(batch_size, num_samples, input_size)
    )
    logvar_q = (
        logvar_q.view(batch_size, -1)
        .unsqueeze(1)
        .expand(batch_size, num_samples, input_size)
    )
    mu_p = (
        mu_p.view(batch_size, -1)
        .unsqueeze(1)
        .expand(batch_size, num_samples, input_size)
    )
    logvar_p = (
        logvar_p.view(batch_size, -1)
        .unsqueeze(1)
        .expand(batch_size, num_samples, input_size)
    )

    # literally compute expectation of log q(z) - log p(z) over samples z ~ q
    pi = torch.tensor(torch.pi)

    # first sample z from q distribution via reparameterization trick
    # z is of shape [batch_size, num_samples, dim]
    std_q = torch.exp(0.5 * logvar_q)
    z = mu_q + torch.randn_like(std_q) * std_q

    # now we compute log q(z) and log p(z) -> [batch_size, num_samples] each at the end after reduction
    log_q = -0.5 * (
        logvar_q + (z - mu_q) ** 2 / torch.exp(logvar_q) + torch.log(2 * pi)
    )
    log_p = -0.5 * (
        logvar_p + (z - mu_p) ** 2 / torch.exp(logvar_p) + torch.log(2 * pi)
    )

    log_q = torch.sum(log_q, dim=-1)
    log_p = torch.sum(log_p, dim=-1)

    # now compute direct log diff
    kl_mc_per_sample = log_q - log_p
    kl_mc = kl_mc_per_sample.mean(-1)

    return kl_mc
