import numpy as np
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, dropout: float = 0.1):
        super().__init__()

        self.lin = nn.Sequential(
            nn.LayerNorm(in_f),
            nn.Linear(in_f, out_f),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.lin(x)

        return x


class Encoder(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()

        self._check_layers(layer_sizes=layer_sizes)

        self.layer_sizes = layer_sizes

        self.encoder = nn.ModuleList()

        for i in range(1, len(layer_sizes)):
            self.encoder.append(
                LinearLayer(in_f=layer_sizes[i - 1], out_f=layer_sizes[i])
            )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x

    @staticmethod
    def _check_layers(layer_sizes: list[int]):
        sizes = np.array(layer_sizes[1:], dtype=int)

        diffs = sizes[:-1] - sizes[1:]

        assert (diffs > 0).all(), "Layers don't have decreasing sizes."


class Decoder(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()

        self._check_layers(layer_sizes=layer_sizes)

        self.layer_sizes = layer_sizes

        self.decoder = nn.ModuleList()

        for i in range(1, len(layer_sizes)):
            self.decoder.append(
                LinearLayer(in_f=layer_sizes[i - 1], out_f=layer_sizes[i])
            )

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)

        return x

    @staticmethod
    def _check_layers(layer_sizes: list[int]):
        sizes = np.array(layer_sizes[:-1], dtype=int)

        diffs = sizes[:-1] - sizes[1:]

        assert (diffs < 0).all(), "Layers don't have increasing sizes."
