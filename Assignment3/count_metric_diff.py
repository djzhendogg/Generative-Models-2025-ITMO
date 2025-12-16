from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch import nn
from tqdm import tqdm
from unet import UNet


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


def load_model(path: str, device: torch.device) -> DenoiseDiffusion:
    """
    Загружает модель из чекпоинта и возвращает готовый экземпляр DenoiseDiffusion.

    :param path: путь к сохранённому .pth файлу
    :param device: устройство (cpu/cuda)
    :return: экземпляр DenoiseDiffusion
    """
    checkpoint = torch.load(path, map_location=device)

    # Восстанавливаем UNet
    model = UNet(image_channels=checkpoint['image_channels'], n_channels=checkpoint['n_channels'],
                 ch_mults=checkpoint['channel_multipliers'], is_attn=checkpoint['is_attention']).to(device)

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])

    # Создаём DenoiseDiffusion
    diffusion = DenoiseDiffusion(eps_model=model, n_steps=checkpoint['n_steps'], device=device)

    return diffusion


model = load_model(
    "/mnt/tank/scratch/edin/Generative-Models-2025-ITMO/Assignment3/zhenya/diffusion/diffusion_model.pth",
    torch.device('cuda'))
print("Model Loaded")


# class CIFAR10Dataset(torchvision.datasets.CIFAR10):
#     def __init__(self, image_size):
#         transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(image_size),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize((0.5, 0.5, 0.5),
#                                              (0.5, 0.5, 0.5)),
#         ])
#         super().__init__("data", train=True, download=True, transform=transform)
#
#     def __getitem__(self, item):
#         return super().__getitem__(item)[0]

# def plot_samples(tensor):
#     # Assuming you have ass tensor of size torch.Size([16, 1, 32, 32])
#     # Convert the tensor to a numpy array
#     images = tensor.reshape(16, 3, 64, 64).numpy()

#     # Reshape the images to be of size (16, 32, 32)

#     # Create a figure with a grid of subplots
#     fig, axes = plt.subplots(nrows=4, ncols=4)

#     # Iterate over the images and plot them on the subplots
#     for i, ax in enumerate(axes.flatten()):
#         ax.imshow(images[i])
#         ax.axis('off')

#     # Show the plot
#     plt.show()
def denorm(img):
    return img.add(1).div(2).clamp(0, 1)


def plot_samples(tensor):
    images = denorm(tensor).reshape(16, 3, 64, 64).numpy()
    images = images.transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

    for i, ax in enumerate(axes.flatten()):
        img = images[i] * 255
        img = img.astype('uint8')
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


class Configs:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # U-Net model for epsilon_theta(x_t, t)
    eps_model: UNet
    # DDPM algorithm
    diffusion: DenoiseDiffusion

    # Number of channels in the image. 3 for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 64
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps T
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 128
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 5

    # Dataset
    # dataset: torch.utils.data.Dataset = CIFAR10Dataset(image_size)
    # Dataloader
    # data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        # Create epsilon_theta(x_t, t) model
        self.eps_model = UNet(image_channels=self.image_channels, n_channels=self.n_channels,
            ch_mults=self.channel_multipliers, is_attn=self.is_attention, ).to(self.device)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(eps_model=self.eps_model, n_steps=self.n_steps, device=self.device, )

        # Create dataloader
        # self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        # self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

    def sample(self):
        with torch.no_grad():
            # [1]
            x = torch.randn((self.n_samples, self.image_channels, self.image_size, self.image_size), device=self.device)

            # Remove noise for T steps
            progress_bar = tqdm(range(self.n_steps))
            for t_ in progress_bar:
                progress_bar.set_description(f"Sampling")
                # t
                t = self.n_steps - t_ - 1
                # [2]
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            plot_samples(x.detach().cpu())

    def sample_one(self):
        with torch.no_grad():
            # [1]
            x = torch.randn((1, self.image_channels, self.image_size, self.image_size), device=self.device)

            # Remove noise for T steps
            for t_ in range(self.n_steps):
                # t
                t = self.n_steps - t_ - 1
                # [2]
                x = self.diffusion.p_sample(x, x.new_full((1,), t, dtype=torch.long))
            return x.detach()


# Create configurations
configs = Configs()

# Initialize
configs.init()

configs.diffusion = model

print("one sample:")
print(configs.sample_one().shape)

npz_path = "/mnt/tank/scratch/edin/Generative-Models-2025-ITMO/Assignment3/zhenya/fid_stats_cifar10_train.npz"
from pytorch_image_generation_metrics import (get_inception_score, get_fid, get_inception_score_and_fid)
from pytorch_image_generation_metrics import ImageDataset
from torch.utils.data import DataLoader, Dataset


class GeneratorDataset(Dataset):
    def __init__(self, config):
        self.config = config  # self.noise_dim = noise_dim

    def __len__(self):
        return 50

    def __getitem__(self, index):
        y = self.config.sample_one()
        yy = denorm(y)[0] * 255
        return yy


dataset = GeneratorDataset(configs)
# create dataloader
loader = DataLoader(dataset, batch_size=50, num_workers=0)
# Inception Score
# IS, IS_std = get_inception_score(loader)
# Frechet Inception Distance
# FID = get_fid(
#     loader, npz_path)
# Inception Score + Frechet Inception Distance
print("Count metrics...")
(IS, IS_std), FID = get_inception_score_and_fid(loader, npz_path)
print("IS: %.3f, IS_std: %.3f, FID: %.3f" % (IS, IS_std, FID))
