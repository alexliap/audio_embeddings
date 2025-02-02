import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam

from audio_embeddings.autoencoder import AutoEncoder


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

    def get_distance(self, array_1: np.ndarray, array_2: np.ndarray) -> float:
        """Cacluate the Euclidean distance between 2 vectors. The second vector
        can have a batch dimension > 1. In order to calculate the distance,
        the latent representations are computed by using the Encoder of the model
        and finally calculating the distance between them. If the second array has
        a batch dimension > 1, then the distance calculated is the average distance
        between array_1 and array_2.

        Args:
            array_1 (np.ndarray): Array 1.
            array_2 (np.ndarray): Array 2.

        Returns:
            float: Euclidean distance
        """
        array_1 = torch.from_numpy(array_1)
        array_2 = torch.from_numpy(array_2)

        self.autoencoder.train(False)

        emb_1 = self.autoencoder.encoder(array_1).detach().numpy()
        emb_2 = self.autoencoder.encoder(array_2).detach().numpy()

        euclidean_distance = np.linalg.norm(emb_1 - emb_2, axis=1).mean()

        return euclidean_distance.item()
