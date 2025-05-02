# %%
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from easydict import EasyDict
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from cfg_utils.args import *


class CFGDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps

        self.lambda_min = -20
        self.lambda_max = 20

        self.device = device

    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l - l_prim)

    def get_lambda(self, t: torch.Tensor):
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)

        # divide by the number of steps to keep uniform
        t = t.float() / self.n_steps

        # move `lambda_min`, `lambda_max` to device as tensors
        lambda_min = torch.tensor(self.lambda_min, device=t.device)
        lambda_max = torch.tensor(self.lambda_max, device=t.device)

        # in this case, t = u, so compute the terms accordingly
        b = torch.arctan(torch.exp(-lambda_max / 2.0))
        a = torch.arctan(torch.exp(-lambda_min / 2.0)) - b

        lambda_t = -2.0 * torch.log(torch.tan(a * t + b))
        lambda_t = lambda_t.view(-1, 1, 1, 1)

        return lambda_t

    def alpha_lambda(self, lambda_t: torch.Tensor):
        # TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)

        var = F.sigmoid(lambda_t)
        return var.sqrt()

    def sigma_lambda(self, lambda_t: torch.Tensor):
        # TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)

        # sigma^2(lambda) = 1 - alpha^2(lambda)
        alpha2_lambda = F.sigmoid(lambda_t)
        var = 1.0 - alpha2_lambda

        return var.sqrt()

    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        # TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)

        alpha_lambda = self.alpha_lambda(lambda_t)
        sigma_lambda = self.sigma_lambda(lambda_t)

        z_lambda_t = alpha_lambda * x + noise * sigma_lambda
        return z_lambda_t

    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        # TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)

        coeff = 1.0 - self.get_exp_ratio(lambda_t, lambda_t_prim)
        sigma2_lambda_t = 1.0 - F.sigmoid(lambda_t)

        var_q = coeff * sigma2_lambda_t
        return var_q.sqrt()

    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        # TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)

        coeff = 1.0 - self.get_exp_ratio(lambda_t, lambda_t_prim)
        sigma2_lambda_t_prim = 1.0 - F.sigmoid(lambda_t_prim)

        var_q_x = coeff * sigma2_lambda_t_prim
        return var_q_x.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(
        self,
        z_lambda_t: torch.Tensor,
        x: torch.Tensor,
        lambda_t: torch.Tensor,
        lambda_t_prim: torch.Tensor,
    ):
        # TODO: Write function that returns mean of the forward process transition distribution according to (4)

        coeff = self.get_exp_ratio(lambda_t, lambda_t_prim)
        alpha_ratio = self.alpha_lambda(lambda_t_prim) / self.alpha_lambda(lambda_t)
        z_coeff = coeff * alpha_ratio

        x_coeff = (1.0 - coeff) * self.alpha_lambda(lambda_t_prim)

        mu = z_lambda_t * z_coeff + x * x_coeff
        return mu

    def var_p_theta(
        self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float = 0.3
    ):
        # TODO: Write function that returns var of the forward process transition distribution according to (4)

        # compute sigma_tilde first
        sigma_tilde = self.sigma_q_x(lambda_t, lambda_t_prim)
        sigma = self.sigma_q(lambda_t, lambda_t_prim)

        var = sigma_tilde ** (2.0 - 2 * v) * sigma ** (2 * v)
        return var

    def p_sample(
        self,
        z_lambda_t: torch.Tensor,
        lambda_t: torch.Tensor,
        lambda_t_prim: torch.Tensor,
        x_t: torch.Tensor,
        set_seed=False,
    ):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4)
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)

        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        std = self.var_p_theta(lambda_t, lambda_t_prim).sqrt().expand_as(mu)

        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample

    ### LOSS
    def loss(
        self,
        x0: torch.Tensor,
        labels: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        set_seed=False,
    ):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)

        # first get lambda_t from t
        lambda_t = self.get_lambda(t)

        # now get z_lambda_t
        z_lambda_t = self.q_sample(x0, lambda_t, noise)

        # pass z_lambda_t through the model
        pred_noise = self.eps_model(z_lambda_t, labels)

        # compute loss
        loss = ((pred_noise - noise) ** 2).sum(dim=dim).mean()
        return loss
