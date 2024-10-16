from dataclasses import dataclass
from skimage import color

import lightning as L
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset    

# Data and training parameters
num_workers: int = 4
max_epochs: int = 10
lr = 1e-3

@dataclass
class Config():
    # Model
    sample_size: int = 1024
    in_channels: int = 1
    out_channels: int = 2
    
    # Data
    dataset: str = "MichaelP84/manga-colorization-dataset"
    train_val_split: float = 0.99
    train_batch_size: int = 32
    

class MangaColorizationDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        self.dataset = load_dataset(self.config.dataset, split="train").with_format("torch")
        self.dataset = self.dataset.map(self.rgb2lab, batched=True, batch_size=32)
    
    def rgb2lab(self, x):
        if x["color_image"].dim() == 3:
            image = x["color_image"].permute(1, 2, 0)
            image = color.rgb2lab(image)
            image = torch.tensor(image).permute(2, 0, 1)[1:]
        else:
            image = x["color_image"].permute(0, 2, 3, 1)
            image = color.rgb2lab(image)
            image = torch.tensor(image).permute(0, 3, 1, 2)[:, 1:]
        return {"color_image": image, "bw_image": x["bw_image"]}

    def setup(self, stage=None):
        n_train = int(len(self.dataset) * self.config.train_val_split)
        n_val = len(self.dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)
    
    
class MangaColorizationModel(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.loss = torch.nn.L1Loss()
        
        self.unet = UNet2DModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels)

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=lr)


    

