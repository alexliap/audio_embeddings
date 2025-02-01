import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam

from orfium_audio.autoencoder import AutoEncoder


class AutoEncoderModel(LightningModule):
    def __init__(self, autoencoder: AutoEncoder | None = None, lr: float | None = None):
        super().__init__()

        self.lr = lr
        self.autoencoder = autoencoder

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        out = self.autoencoder.forward(x)
        train_loss = F.mse_loss(out, x)
        self.log("train_mse", train_loss, prog_bar=True)

        return {"loss": train_loss}

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        out = self.autoencoder.forward(x)
        val_loss = F.mse_loss(out, x)

        self.log("val_mse", val_loss, prog_bar=True)

        return {"loss": val_loss}

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr)

        return optim

    def get_distance(self, array_1: np.ndarray, array_2: np.ndarray):
        array_1 = torch.from_numpy(array_1)
        array_2 = torch.from_numpy(array_2)

        self.autoencoder.train(False)

        emb_1 = self.autoencoder.encoder(array_1)
        emb_2 = self.autoencoder.encoder(array_2)

        euclidean_distance = torch.sum((emb_1 - emb_2) ** 2)

        return euclidean_distance.item()
