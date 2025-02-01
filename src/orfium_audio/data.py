from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, csv_path: Path, drop_columns: list | None = None):
        tabular_dataset = pl.scan_csv(csv_path.resolve())

        if drop_columns is not None:
            tabular_dataset = tabular_dataset.drop(drop_columns)

        self.tensor_data = TensorDataset(
            torch.from_numpy(tabular_dataset.collect().to_numpy().astype(np.float32))
        )

    def get_loader(self, batch_size: int = 512) -> DataLoader:
        return DataLoader(dataset=self.tensor_data, batch_size=batch_size)
