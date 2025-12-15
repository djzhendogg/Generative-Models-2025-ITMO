import torch
import torchvision.utils as vutils
from PIL import Image

from arh import *
from unet import UNet

def denorm(img):
    return img.add(1).div(2).clamp(0,1)

def save_samples(tensor: torch.Tensor, epoch: int, nrow: int = 4, padding: int = 2):
    """
    Сохраняет тензор изображений в виде сетки на локальный диск.

    :param epoch: номер эпохи
    :param tensor: тензор изображений [B, C, H, W] (на GPU или CPU)
    :param path: путь для сохранения изображения (например, 'samples/epoch_10.png')
    :param nrow: количество изображений в ряду сетки
    :param padding: отступы между изображениями в пикселях
    :param normalize: если True — нормализует значения в диапазон value_range
    :param value_range: минимальный и максимальный значения для нормализации (по умолчанию 0–1)
    """
    # Переносим на CPU и отвязываем от графов
    images = denorm(tensor).detach().cpu()

    # Создаём сетку (как в torchvision.utils.make_grid)
    grid = vutils.make_grid(images, nrow=nrow, padding=padding, normalize=False)

    # Преобразуем из тензора [C, H, W] в PIL Image
    grid_np = grid.permute(1, 2, 0).numpy()  # [H, W, C]
    grid_np = (grid_np * 255).astype('uint8')  # в диапазон 0–255
    image = Image.fromarray(grid_np)

    # Сохраняем на диск
    image.save(f"./images/fake_image_new_{epoch}.png")


def save_model(configs, path: str):
    """
    Сохраняет модель и optimizer в указанный путь.

    :param configs: экземпляр Configs (с обученной моделью)
    :param path: путь к файлу (.pth или .pt)
    """
    torch.save({
        'model_state_dict': configs.eps_model.state_dict(),
        'optimizer_state_dict': configs.optimizer.state_dict(),
        'n_steps': configs.n_steps,
        'image_channels': configs.image_channels,
        'n_channels': configs.n_channels,
        'channel_multipliers': configs.channel_multipliers,
        'is_attention': configs.is_attention,
    }, path)


def load_model(path: str, device: torch.device) -> DenoiseDiffusion:
    """
    Загружает модель из чекпоинта и возвращает готовый экземпляр DenoiseDiffusion.

    :param path: путь к сохранённому .pth файлу
    :param device: устройство (cpu/cuda)
    :return: экземпляр DenoiseDiffusion
    """
    checkpoint = torch.load(path, map_location=device)

    # Восстанавливаем UNet
    model = UNet(
        image_channels=checkpoint['image_channels'],
        n_channels=checkpoint['n_channels'],
        ch_mults=checkpoint['channel_multipliers'],
        is_attn=checkpoint['is_attention']
    ).to(device)

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])

    # Создаём DenoiseDiffusion
    diffusion = DenoiseDiffusion(
        eps_model=model,
        n_steps=checkpoint['n_steps'],
        device=device
    )

    return diffusion
