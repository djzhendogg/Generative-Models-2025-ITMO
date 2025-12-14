import os

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from arh import *
from config import CFG


if not os.path.exists(CFG.sample_dir):
    os.makedirs(CFG.sample_dir)

cifar_dataset = CIFAR10(
    root=CFG.dataroot,
    download=CFG.download,
    transform=transforms.Compose(
        [
            transforms.Resize([CFG.image_size, CFG.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
# unnormalization image from range (-1)-1 to range 0-1 to display it
def denorm(img):
    return img.add(1).div(2).clamp(0,1)

# define the dataloader
data_loader = DataLoader(cifar_dataset, CFG.batch_size, shuffle=True)

G = Generator(CFG.nc, CFG.nz, CFG.ngf)
D = Discriminator(CFG.nc, CFG.ndf)

criterion = nn.BCELoss()

g_optimizer = torch.optim.Adam(G.parameters(), lr=CFG.lr)
d_optimizer = torch.optim.Adam(D.parameters(), lr=CFG.lr)


def reset_grad():
    ## reset gradient for optimizer of generator and discrimator
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()


LABEL_SMOOTH = 0.95
def train_discriminator(images):
    batch_size = images.size(0)
    images = images.view(images.size(0), CFG.nc, CFG.image_size, CFG.image_size).to(device)

    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.empty(batch_size, 1, device=device).uniform_(0.7, 1.0)
    fake_labels = torch.empty(batch_size, 1, device=device).uniform_(0.0, 0.3)

    flip_mask_real = torch.rand(batch_size, 1, device=device) < 0.05
    real_labels[flip_mask_real] = torch.empty_like(real_labels[flip_mask_real]).uniform_(0.0, 0.3)

    flip_mask_fake = torch.rand(batch_size, 1, device=device) < 0.05
    fake_labels[flip_mask_fake] = torch.empty_like(fake_labels[flip_mask_fake]).uniform_(0.7, 1.0)


    outputs = D(images)
    # Loss for real images
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images

    z = torch.randn(batch_size, CFG.nz, 1, 1).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Sum losses
    d_loss = d_loss_real + d_loss_fake

    # Reset gradients
    reset_grad()

    # Compute gradients

    # Adjust the parameters using backprop
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score

def train_generator():
    # Generate fake images and calculate loss
    # z = torch.randn(batch_size, latent_size).to(device)
    z = np.random.normal(0, 1, (batch_size, CFG.nz, 1, 1))
    noise = 0.005*np.random.uniform()*np.amax(z)
    z = z.astype('float64') + noise*np.random.normal(size=z.shape)
    z = torch.Tensor(z).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)

    # calculate the generator loss
    outputs = D(fake_images)
    g_loss = criterion(outputs, labels)

    # Reset gradients
    reset_grad()

    # Backprop and optimize
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_images

def save_fake_images(index):
    # sample_vectors = torch.randn(batch_size, latent_size).to(device)
    z = np.random.normal(0, 1, (CFG.batch_size, CFG.nz, 1, 1))
    noise = 0.005*np.random.uniform()*np.amax(z)
    z = z.astype('float64') + noise*np.random.normal(size=z.shape)
    z = torch.Tensor(z).to(device)
    fake_images = G(z)
    fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(CFG.sample_dir, fake_fname), nrow=10)

device = CFG.device
num_epochs = CFG.num_epochs
batch_size = CFG.batch_size

total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []
G.to(device)
D.to(device)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Load a batch & transform to vectors
        batch_size = images.size(0)
        images = images.reshape(batch_size, -1).to(device)

        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(images)

        # Train the generator
        g_loss, fake_images = train_generator()

        # Inspect the losses
        if (i+1) % 200 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))
    # Sample and save images
    save_fake_images(epoch+1)

np.save('d_losses.npy', np.array(d_losses))
np.save('g_losses.npy', np.array(g_losses))
np.save('real_scores.npy', np.array(real_scores))
np.save('fake_scores.npy', np.array(fake_scores))

torch.save(G, "generator.pkl")
model = torch.load("generator.pkl", weights_only=False)
model.eval()
print("Model sucsesfully downloaded")
