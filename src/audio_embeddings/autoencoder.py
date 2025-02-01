import torch.nn as nn

from orfium_audio.linear import Decoder, Encoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        self._check_modules(encoder, decoder)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    @staticmethod
    def _check_modules(encoder: Encoder, decoder: Decoder):
        enc_layers = encoder.layer_sizes
        dec_layers = decoder.layer_sizes

        assert enc_layers[-1] == dec_layers[0], (
            "Output size of encoder is not equal with input of decoder."
        )

        assert enc_layers[0] == dec_layers[-1], (
            "Input and ouput sizes of Autoencoder are not equal."
        )
