from dataclasses import dataclass
from skimage import color

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
import torchvision.transforms as T
from diffusers.models.unets.unet_2d import UNet2DModel
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset

# Data and training parameters
num_workers: int = 4
max_epochs: int = 10
lr = 1e-3
wandb = False

@dataclass
class Config():
    # Model
    sample_size: int = 1024
    in_channels: int = 1
    out_channels: int = 2
    timestep: int = 1
    
    # Data
    dataset: str = "MichaelP84/manga-colorization-dataset"
    train_val_split: float = 0.99
    train_batch_size: int = 1
    streaming = False
    

class MangaColorizationDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        self.dataset = load_dataset(self.config.dataset, split="train", streaming=self.config.streaming).with_format("torch")

    def setup(self, stage=None):
        n_train = int(len(self.dataset) * self.config.train_val_split)
        n_val = len(self.dataset) - n_train
        self.train_dataset, self.val_dataset = random_split(self.dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, num_workers=num_workers, collate_fn=self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=num_workers, collate_fn=self._collate_fn)
    
    def _collate_fn(self, batch):
        # Si el batch es una lista de diccionarios, ajusta el acceso
        color_images = torch.stack([item["color_image"] for item in batch])
        bw_images = torch.stack([item["bw_image"] for item in batch],)
        
        # Convertir las imágenes de RGB a LAB y ajustar las dimensiones
        y = color.rgb2lab(color_images.permute(0, 2, 3, 1).numpy())
        y = torch.tensor(y, dtype=torch.float).permute(0, 3, 1, 2)[:, 1:]  # Excluir canal L
        
        return bw_images.to(dtype=y.dtype), y
    
    
class MangaColorizationModel(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        
        self.unet = UNet2DModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels)

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x, config.timestep)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.unet(x, config.timestep)
        if wandb:
            logger.log_image("color", torch.cat([x, y_hat], dim=1))
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)


if __name__ == "__main__":
    config = Config()
    logger = WandbLogger(project="manga-colorization", log_model='all') if wandb else None
    dm = MangaColorizationDataModule(config)
    model = MangaColorizationModel(config)
    trainer = L.Trainer(max_epochs=max_epochs, 
                        logger=logger,
                        accelerator="auto",
                        overfit_batches=1,
                        num_sanity_val_steps=0)
    trainer.fit(model, dm)
    

