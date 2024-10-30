import os
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage import color

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from datasets import load_dataset

# Data and training parameters
num_workers: int = 4
max_epochs: int = 50
lr = 1e-4
save_images = True
wandb = True

torch.set_float32_matmul_precision("high")

@dataclass
class Config():
    # Model
    sample_size: int = 512
    in_channels: int = 1
    out_channels: int = 2
    timestep: int = 1
    layers_per_block= 2  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 256, 512)  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )

    # Data
    dataset: str = "MichaelP84/manga-colorization-dataset"
    train_val_split: float = 0.99
    train_batch_size: int = 4
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
        bw_images = torch.stack([item["bw_image"] for item in batch])
        
        color_images = F.interpolate(color_images, size=(self.config.sample_size, self.config.sample_size), mode="bilinear", align_corners=False)
        bw_images = F.interpolate(bw_images, size=(self.config.sample_size, self.config.sample_size), mode="bilinear", align_corners=False)
        
        # Convertir las imágenes de RGB a LAB y ajustar las dimensiones
        color_images = color.rgb2lab(color_images.permute(0, 2, 3, 1).numpy())
        color_images = torch.tensor(color_images, dtype=torch.float).permute(0, 3, 1, 2)[:, 1:]  # Excluir canal L
        
        return bw_images.to(dtype=color_images.dtype), color_images
    
    
class MangaColorizationModel(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        
        self.unet = UNet2DModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        assert x.size(-1) == config.sample_size, f"Expected sample size {config.sample_size}, got {x.size(-1)}"
        
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu' ,dtype=torch.bfloat16):
            y_hat = self.unet(x, config.timestep).sample
            
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu' ,dtype=torch.bfloat16):
            y_hat = self.unet(x, config.timestep).sample
        
        # Save the image locally
        if save_images:
            color_image = torch.cat([x, y_hat], dim=1)
            color_image = color_image.permute(0, 2, 3, 1).cpu()  # Convert to numpy array
            color_image = color.lab2rgb(color_image)  # Convert to RGB
            if not os.path.exists("output_images"):
                os.makedirs("output_images")
            for i, img in enumerate(color_image):
                img_path = f"output_images/validation_image_{batch_idx}_{i}.png"
                plt.imsave(img_path, img.squeeze())
                if wandb:
                    logger.log_image("color", [img.squeeze()])
        
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)


if __name__ == "__main__":
    config = Config()
    logger = WandbLogger(project="manga-colorization", 
                         log_model='all',
                         name = "test_dev") if wandb else None
    dm = MangaColorizationDataModule(config)
    model = MangaColorizationModel(config)
    trainer = L.Trainer(fast_dev_run = False,
                        logger=logger,
                        accelerator="auto",
                        max_epochs=max_epochs
                        )
    trainer.fit(model, dm)