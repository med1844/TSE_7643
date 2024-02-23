from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data.dataloader import Dataset, DataLoader
import torchaudio
from glob import glob
import pandas as pd
import os
import io
from tqdm import tqdm
from functools import reduce
from operator import or_
from librosa.filters import mel as librosa_mel_fn
import random
import torch.utils.data
from dataclasses import dataclass
from traits import SerdeJson, Json
from audio_commons import read_wav_at_fs


class SpeakerAudioProvider:
    """
    A trait for all datasets that satisfies two properties:
    - Has speakers
    - Each speaker has some speech utterances
    """

    @abstractmethod
    def get_speaker_list(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_speaker_files(self, name: str) -> Iterable[torch.Tensor]:
        """given a speaker id string, returns a list of raw wav array"""
        pass


class GenshinDataset(SpeakerAudioProvider):
    def __init__(self, data: Dict[str, List[torch.Tensor]]) -> None:
        self.data = data

    def get_speaker_list(self) -> Iterable[str]:
        return self.data.keys()

    def get_speaker_files(self, name: str) -> Iterable[torch.Tensor]:
        return self.data.get(name, [])

    @staticmethod
    def process_df(df: pd.DataFrame):
        # WARNING you should not call this method and view it as private
        data = {}
        for _i, row in df.iterrows():
            audio: bytes = row["audio"]["bytes"]
            name: str = row["npcName"]
            if name:
                data.setdefault(name, []).append(
                    torchaudio.functional.resample(
                        *torchaudio.load(io.BytesIO(audio), backend="ffmpeg"), 48000
                    )  # we know the data is 48000, but just to ensure
                )
        return data

    @classmethod
    def from_parquets(cls, parquet_folder: str) -> "GenshinDataset":
        # after `git clone https://huggingface.co/datasets/hanamizuki-ai/genshin-voice-v3.5-mandarin`
        # fed the folder path to the git clone result into this function
        # NOTE this might take a long time to read (1-2 min on mech disk)
        dfs = [
            pd.read_parquet(filename)
            for filename in tqdm(
                glob(os.path.join(parquet_folder, "*.parquet")), desc="parquet file"
            )
        ]

        with ProcessPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(cls.process_df, dfs),
                    total=len(dfs),
                    desc="Processing files",
                )
            )
            return cls(reduce(or_, results))


class SpeechDataset(Dataset):
    """This dataset stores a bunch of speech, regardless of speaker"""

    def __init__(self, audios: List[torch.Tensor]) -> None:
        super().__init__()
        self.audios = audios

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.audios[index]

    @classmethod
    def from_folder(cls, folder_path: str, target_fs=48000) -> "SpeechDataset":
        return cls(
            list(
                read_wav_at_fs(target_fs, filename)
                for filename in glob(os.path.join(folder_path, "*.wav"))
            )
        )

    @classmethod
    def from_speaker_audio_provider(
        cls, provider: SpeakerAudioProvider
    ) -> "SpeechDataset":
        audios = []
        for spk in provider.get_speaker_list():
            audios.extend(list(provider.get_speaker_files(spk)))
        return cls(audios)


def pad_seq_n_stack(wavs: Iterable[torch.Tensor], target_len: int) -> torch.Tensor:
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


def speech_dataloader_collate_fn(batch: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        batch: list of audio tensor, i.e. List[SpeechDataset.__getitem__ return type]
               each item has size [1 x T]
    Returns:
        (
            batch_padded_wav: B x max(T), Tensor
        )
    """
    pad_length = max(mixed.shape[-1] for mixed, _ in batch)
    return pad_seq_n_stack(batch, pad_length)


class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=speech_dataloader_collate_fn, *args, **kwargs)


class TSEDataset(Dataset):
    def __init__(self, provider: SpeakerAudioProvider) -> None:
        super().__init__()
        # TODO how should we turn speaker-audio pairs into TSE dataset?
        # We need:
        #   - __len__
        #   - a way for dataloader to be splitted into train, eval & test, audio doesn't overlap
        #   - might need the behavior to be deterministic
        raise NotImplementedError


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
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


@dataclass
class MelArgs(SerdeJson):
    segment_size: int
    n_fft: int
    num_mels: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: Optional[int] = None

    def to_json(self) -> Json:
        return {
            "segment_size": self.segment_size,
            "n_fft": self.n_fft,
            "num_mels": self.num_mels,
            "hop_size": self.hop_size,
            "win_size": self.win_size,
            "sampling_rate": self.sampling_rate,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    @classmethod
    def from_json(cls, obj: Json) -> "MelArgs":
        return cls(
            segment_size=obj["segment_size"],
            n_fft=obj["n_fft"],
            num_mels=obj["num_mels"],
            hop_size=obj["hop_size"],
            win_size=obj["win_size"],
            sampling_rate=obj["sampling_rate"],
            fmin=obj["fmin"],
            fmax=obj["fmax"],
        )


MelDatasetOutput = Tuple[torch.Tensor, torch.Tensor]


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audios: List[torch.Tensor], args: MelArgs):
        self.audios = [
            a
            for a in audios
            if 5 * args.sampling_rate <= a.shape[-1] < 15 * args.sampling_rate
        ]
        self.segment_size = args.segment_size
        self.sampling_rate = args.sampling_rate
        self.n_fft = args.n_fft
        self.num_mels = args.num_mels
        self.hop_size = args.hop_size
        self.win_size = args.win_size
        self.fmin = args.fmin
        self.fmax = args.fmax

    def __getitem__(self, index: int) -> MelDatasetOutput:
        audio = self.audios[index]
        audio = torch.FloatTensor(audio)

        if audio.size(-1) >= self.segment_size:
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[..., audio_start : audio_start + self.segment_size]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_size - audio.size(1)), "constant"
            )

        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax,
            center=False,
        )

        return (mel.squeeze(), audio.squeeze(0))

    def __len__(self):
        return len(self.audios)

    @classmethod
    def from_speech_dataset(cls, dataset: SpeechDataset, args: MelArgs) -> "MelDataset":
        return cls(dataset.audios, args)


if __name__ == "__main__":
    import sys

    GenshinDataset.from_parquets(sys.argv[1])
