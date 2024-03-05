import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio


def melspec_torchaudio(
    audio, sr=22000, n_mels=32, n_fft=1024, hop_length=176, win_length=176 * 2
):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad=0,
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)
    return mel(audio)


def melspec_librosa(
    audio, sample_rate=22000, n_mels=32, n_fft=1024, hop_length=176, win_length=176 * 2
):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=n_mels,
        norm="slaney",
        htk=True,
    )
    # mel = librosa.power_to_db(mel, ref=np.max)
    return mel


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(
        librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
