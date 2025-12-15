from typing import List

import numpy as np
import torch
import torch.utils.data
import torchvision

from arh import DenoiseDiffusion
from common import save_samples, save_model, load_model
from unet import UNet


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5)),
        ])
        super().__init__("data", train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


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
    is_attention: List[bool] = [False, False, False, True]

    # Number of time steps T
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 128
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 102

    losses: List[float] = []

    # Dataset
    dataset: torch.utils.data.Dataset = CIFAR10Dataset(image_size)
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        # Create epsilon_theta(x_t, t) model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        # Create DDPM class
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

    def sample(self, epoch):
        with torch.no_grad():
            # [1]
            x = torch.randn((self.n_samples, self.image_channels, self.image_size, self.image_size), device=self.device)

            # Remove noise for T steps
            progress_bar = range(self.n_steps)
            for t_ in progress_bar:
                # t
                t = self.n_steps - t_ - 1
                # [2]
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            if epoch % 50 == 0:
                save_samples(x.detach().cpu(), epoch)

    def train(self, epoch):
        # Iterate through the dataset
        progress_bar = self.data_loader
        losses_list = []
        for data in progress_bar:
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()

            # Calculate loss
            loss = self.diffusion.loss(data)
            losses_list.append(loss)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
        self.losses.append(sum(losses_list) / len(losses_list))
        print('Epoch [{}/{}], loss: {:.4f}'.format(epoch, self.epochs, loss.item()))

    def run(self):
        for epoch in range(self.epochs):
            # Train the model
            self.train(epoch)
            # Sample some images
            print("Sampling")
            self.sample(epoch)
            if (epoch + 1) % 10 == 0:
                save_model(configs, "diffusion_model.pth")
                print(f"Model saved success on epoch: {epoch + 1}")

# Create configurations
configs = Configs()

# Initialize
configs.init()
print(f"Device: {configs.device}")
# Start and run the training loop
configs.run()

# Save model
save_model(configs, "diffusion_model.pth")

# Load Model
diffusion = load_model("diffusion_model.pth", configs.device)
print("Model saved success")

np.save('g_losses.npy', np.array(configs.losses))
