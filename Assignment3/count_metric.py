from pytorch_image_generation_metrics import (
    get_inception_score,
    get_fid,
    get_inception_score_and_fid
)
import torch
from pytorch_image_generation_metrics import ImageDataset
from torch.utils.data import DataLoader, Dataset
device = "cuda"
npz_path = "/mnt/tank/scratch/edin/Generative-Models-2025-ITMO/Assignment3/zhenya/fid_stats_cifar10_train.npz"
generatore_path = "/mnt/tank/scratch/edin/Generative-Models-2025-ITMO/Assignment3/danya/generator.pkl"
G_loaded = torch.load(generatore_path, weights_only=False, map_location=torch.device('cuda'))
G_loaded.eval()
print("Model Loaded")
z = torch.randn(1, 100, 1, 1).to(device)
print(G_loaded(z)[0].shape)
# create the Generator Dataset. returns the image, generated from the Generator network
class GeneratorDataset(Dataset):
    def __init__(self, G, noise_dim):
        self.G = G
        # self.noise_dim = noise_dim

    def __len__(self):
        return 500

    def __getitem__(self, index):
        z = torch.randn(1, 100, 1, 1).to(device)
        return self.G(z)[0]

# define dataset
dataset = GeneratorDataset(G_loaded, noise_dim=128)
#create dataloader
loader = DataLoader(dataset, batch_size=50, num_workers=0)
# Inception Score
# IS, IS_std = get_inception_score(loader)
# Frechet Inception Distance
# FID = get_fid(
#     loader, npz_path)
# Inception Score + Frechet Inception Distance
print("Count metrics...")
(IS, IS_std), FID = get_inception_score_and_fid(
    loader, npz_path)
print("IS: %.3f, IS_std: %.3f, FID: %.3f" % (IS, IS_std, FID))