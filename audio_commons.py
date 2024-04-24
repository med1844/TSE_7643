from typing import BinaryIO, Tuple, Union, Optional, List, Iterator
import torch
import torchaudio
from PIL import Image
import matplotlib.pyplot as plt
import io
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import pyloudnorm
from pydantic import BaseModel
from dataclasses import dataclass


def amplitude_to_db_torch(magnitude: torch.Tensor, ref=1.0, amin=1e-5, top_db=80.0):
    power = torch.square(magnitude)

    log_spec = 10.0 * torch.log10(torch.maximum(torch.tensor(amin**2), power))
    log_spec -= 10.0 * torch.log10(torch.tensor(max(amin**2, ref**2)))
    log_spec = torch.clamp(log_spec, min=log_spec.max() - top_db)
    log_spec = torch.clamp(log_spec, min=-100)

    return log_spec


def db_to_amplitude_torch(db: torch.Tensor, ref=1.0):
    power = ref**2 * torch.pow(10, db / 10.0)
    return torch.sqrt(power)


class STFTArgs(BaseModel):
    n_fft: int = 2048
    hop_size: int = 240
    win_size: int = 1200

    @property
    def window(self) -> torch.Tensor:
        return torch.hann_window(self.win_size)


@dataclass
class Spectrogram:
    spec: torch.Tensor
    phase: torch.Tensor

    @classmethod
    def from_wav(cls, stft_args: STFTArgs, wav: torch.Tensor):
        window = stft_args.window.to(wav.device)
        wav_stft = torch.stft(
            wav,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=window,
            return_complex=True,
        )
        wav_stft = torch.einsum("bnt -> btn", wav_stft)
        wav_mag = amplitude_to_db_torch(wav_stft.abs())
        wav_phase = wav_stft.angle()
        return cls(wav_mag, wav_phase)

    def to_wav(self, stft_args: STFTArgs) -> torch.Tensor:
        wav_mag = db_to_amplitude_torch(self.spec)
        spectrum = torch.einsum("btn -> bnt", torch.polar(wav_mag, self.phase))
        wav = torch.istft(
            spectrum,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=stft_args.window.to(self.spec.device),
        )
        return wav

    def plot_to_tensor(self) -> torch.Tensor:
        # assume it's batched...
        img = plot_spectrogram(self.spec.detach().cpu().numpy()[0].T)
        img = img.convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.spec, self.phase))


def read_wav_at_fs(
    target_fs: Optional[int], filename: Union[BinaryIO, str]
) -> Tuple[torch.Tensor, int]:
    y, fs = torchaudio.load(filename)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if target_fs is not None and fs != target_fs:
        y = torchaudio.functional.resample(y, fs, target_fs)
        fs = target_fs
    return y, fs


def plot_spectrogram(spectrogram: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="magma",
    )

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


def normalize_loudness(y: np.ndarray, fs: int, target_loudness=-30) -> np.ndarray:
    meter = pyloudnorm.Meter(fs)
    loudness = meter.integrated_loudness(y)
    normalized = pyloudnorm.normalize.loudness(y, loudness, target_loudness)
    return normalized


def normalize_loudness_torch(
    y: torch.Tensor, fs: int, target_loudness=-30
) -> torch.Tensor:
    return torch.tensor(normalize_loudness(y.numpy().T, fs, target_loudness).T)


def pad_seq_n_stack(wavs: List[torch.Tensor], target_len: int) -> torch.Tensor:
    """
    Args:
        wavs: list of 1 x T Tensor, T may vary.
        target_len: assert to be max T in that varying 1 x T tensor list.
    Returns:
        result: B x target_len Tensor
    """
    padded_wavs = [
        torch.cat([wav, torch.zeros(target_len - len(wav))])
        for wav in map(lambda x: x[0], wavs)
    ]
    return torch.stack(padded_wavs)


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
