from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from orfium_audio import AutoEncoder, AutoEncoderModel, Dataset, Decoder, Encoder

train_loader = Dataset(
    Path("tabular_data/cover_analysis_tabular.csv"), ["work", "performance"]
).get_loader(batch_size=512)
val_loader = Dataset(
    Path("tabular_data/benchmark_tabular.csv"), ["work", "performance"]
).get_loader(batch_size=512)

enc = Encoder([52, 300, 200, 100, 50])
dec = Decoder([52, 300, 200, 100, 50][::-1])

module = AutoEncoder(encoder=enc, decoder=dec)

model = AutoEncoderModel(autoencoder=module, lr=1e-3)

model = torch.compile(model)

trainer = Trainer(
    accelerator="cpu",
    max_epochs=20,
    log_every_n_steps=1,
    precision="32-true",
    callbacks=[
        ModelCheckpoint(
            monitor="val_mse",
            save_weights_only=True,
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            dirpath="lightning_logs/model",
        ),
        EarlyStopping(
            monitor="val_mse",
            min_delta=3,
            patience=50,
            mode="min",
            strict=True,
            verbose=True,
        ),
    ],
)

if __name__ == "__main__":
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
