import logging

import torch
import torchaudio
from torch import nn


class MFCCExtractor(nn.Module):
    """
    Input shape: (batch, time)
    Output shape: (batch, out_dim, n_mfcc)
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        out_dim: int,
        win_length: int,
        n_fft: int | None = None,
        normalize: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.out_dim = out_dim
        self.win_length = win_length
        self.hop_length = win_length // 2  # 50% overlap
        self.n_fft = n_fft if n_fft else win_length
        self.T = self._get_extractor()
        self.normalize = normalize
        self.__running_for_first_time = True

    def _get_extractor(self):
        return torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = x / 32768.0
        x = self.T(x).transpose(1, 2)
        if self.out_dim != x.shape[1]:
            if self.__running_for_first_time:
                logging.warning(
                    f"MFCCExtractor: Got tensor shape {x.shape}, resizing to {self.out_dim} using bilinear interpolation"
                )
                self.__running_for_first_time = False
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), size=(self.out_dim, self.n_mfcc), mode="bilinear"
            ).squeeze(1)
        return x
