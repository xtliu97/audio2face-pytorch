from typing import TypedDict, List, Mapping, Literal, Tuple
from typing_extensions import Unpack


import torch
import torchaudio
import numpy as np


class BaseExtractor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __inner_init__(
        self,
        feature_type: Literal["mfcc", "mel"],
        sample_rate: int,
        n_feature: int,
        n_fft: int | None = None,
        *,
        win_length: int,
        hop_length: int,
        normalize: bool = False,
    ):
        assert feature_type in ["mfcc", "mel"], "Invalid feature type"
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.n_feature = n_feature
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft if n_fft else win_length
        self.extractor = self._get_extractor(self)
        self.normalize = normalize

    def __repr__(self) -> str:
        return f"""FeatureExtractor(
    feature_type={self.feature_type},
    sample_rate={self.sample_rate},
    n_feature={self.n_feature},
    n_fft={self.n_fft},
    win_length={self.win_length},
    hop_length={self.hop_length},
    normalize={self.normalize},
)
"""

    @staticmethod
    def _get_extractor(self: "BaseExtractor"):
        raise NotImplementedError

    def __call__(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        if self.normalize:
            audio = audio / 32768.0
        return self.extractor(audio)


class MelSpectrogramExtractor(BaseExtractor):
    def __init__(
        self,
        sample_rate: int,
        n_mel: int,
        win_length: int,
        hop_length: int,
        n_fft: int | None = None,
        normalize: bool = False,
    ):
        self.__inner_init__(
            "mel",
            sample_rate,
            n_mel,
            n_fft,
            win_length=win_length,
            hop_length=hop_length,
            normalize=normalize,
        )

    @staticmethod
    def _get_extractor(self):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_feature,
            center=True,
            power=2.0,
            norm="slaney",
            mel_scale="htk",
        )


class MFCCExtractor(BaseExtractor):
    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        win_length: int,
        hop_length: int,
        n_fft: int | None = None,
        normalize: bool = False,
    ):
        self.__inner_init__(
            "mfcc",
            sample_rate,
            n_mfcc,
            n_fft,
            win_length=win_length,
            hop_length=hop_length,
            normalize=normalize,
        )

    @staticmethod
    def _get_extractor(self):
        return torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_feature,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length},
        )
