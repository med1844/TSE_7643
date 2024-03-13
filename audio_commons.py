from typing import BinaryIO, Union, Optional
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
import matplotlib.pyplot as plt
import io
from librosa.filters import mel as librosa_mel_fn
import numpy as np


def read_wav_at_fs(target_fs: int, filename: Union[BinaryIO, str]) -> torch.Tensor:
    y, fs = torchaudio.load(filename)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if fs != target_fs:
        y = torchaudio.functional.resample(y, fs, target_fs)
    return y


def plot_spectrogram(spectrogram: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: Optional[int],
    center=False,
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_mel_torch(
    y: torch.Tensor,
    sr: int,
    num_mel=128,
    hop_size=512,
    win_size=2048,
    fft_size=2048,
    fmin=40,
    fmax=16000,
):
    with torch.no_grad():
        mel_torch = (
            mel_spectrogram(
                y.unsqueeze(0),
                fft_size,
                num_mel,
                sr,
                hop_size,
                win_size,
                fmin,
                fmax,
            )
            .squeeze(0)
            .T
        )
        return mel_torch.cpu().numpy()


if __name__ == "__main__":
    import sys

    img = plot_spectrogram(
        mel_spectrogram(
            read_wav_at_fs(44100, sys.argv[1]),
            2048,
            128,
            44100,
            512,
            2048,
            40,
            16000,
        )
        .numpy()
        .squeeze()
    )
    img.show()
