from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for t and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * eps_model - epsilon_theta(x_t, t) model
        * n_steps - t
        * device - the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # Create beta_1 ... beta_T linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # alpha_t = 1 - beta_t
        self.alpha = (1.0 - self.beta).to(device)
        # [1]
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        # T
        self.n_steps = n_steps
        # sigma^2 = beta
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get q(x_t|x_0) distribution

        [2]
        """
        # [3]
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # [4]
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        Sample from q(x_t|x_0)

        [5]
        """

        # [6]
        if eps is None:
            eps = torch.randn_like(x0)

        # get q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t)
        # Sample from q(x_t|x_0)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from p_theta(x_{t-1}|x_t)

        [7]
        """

        # epsilon_theta(x_t, t)
        eps_theta = self.eps_model(xt, t)
        # [8]
        alpha_bar = gather(self.alpha_bar, t)
        # alpha_t
        alpha = gather(self.alpha, t)
        beta = gather(self.beta, t)
        # [9]
        eps_coef = beta / torch.sqrt(1.0 - alpha_bar)
        # [10]
        mean = (1 / torch.sqrt(alpha)) * (xt - eps_coef * eps_theta)
        # sigma^2
        var = gather(self.sigma2, t)

        # [11]
        eps = torch.randn_like(xt)
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Simplified Loss

        [12]
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random t for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # [13]
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample x_t for q(x_t|x_0)
        xt = self.q_sample(x0, t, eps=noise)
        # [14]
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(eps_theta, noise)
