import torch.nn as nn
from spectral_norm_code import SpectralNorm

# Generator
class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # ConvTranspose2d nz ngf * 8
            # BatchNorm2d
            # ReLU
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # ConvTranspose2d ngf * 8 ngf * 4
            # BatchNorm2d
            # ReLU
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            # ConvTranspose2d ngf * 4 ngf * 2
            # BatchNorm2d
            # ReLU
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # ConvTranspose2d ngf * 2 ngf
            # BatchNorm2d
            # ReLU
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # ConvTranspose2d ngf  nc
            # Tanh
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # Используем SpectralNorm, LeakyReLU, BatchNorm и Dropout
        self.model = nn.Sequential(
            # SpectralNorm Conv2d
            # LeakyReLU
            SpectralNorm(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # SpectralNorm Conv2d
            # BatchNorm2d
            # LeakyReLU
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # SpectralNorm Conv2d
            # BatchNorm2d
            # LeakyReLU
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # SpectralNorm Conv2d
            # BatchNorm2d
            # LeakyReLU
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # SpectralNorm Conv2d
            # Sigmoid
            SpectralNorm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input).view(input.size(0), -1)

