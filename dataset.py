from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Iterator, Generic
import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import torchaudio
from glob import glob
import pandas as pd
import os
import io
from tqdm import tqdm
import random
import torch.utils.data
from dataclasses import dataclass
from traits import SerdeJson, Json
from audio_commons import read_wav_at_fs, mel_spectrogram, normalize_loudness_torch
from pathlib import Path
import pickle


T = TypeVar("T")


# trait LazyLoad<T> {
#   fn load(&self) -> T;
# }
class LazyLoadable(Generic[T]):
    @abstractmethod
    def load(self) -> T:
        pass


@dataclass
class Audio:
    wav: torch.Tensor
    fs: int  # frequency of sampling = sample rate

    @property
    def len(self) -> int:
        return self.wav.shape[-1]

    @property
    def len_in_s(self) -> float:
        return self.len / self.fs


class LoadedAudio(LazyLoadable):
    def __init__(self, au: Audio):
        self.au = au

    def load(self) -> Audio:
        return self.au


class LazyLoadAudio(LazyLoadable):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> Audio:
        wav, fs = read_wav_at_fs(None, self.path)
        return Audio(wav, fs)


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
    def get_speaker_files(self, name: str) -> Iterable[LazyLoadable[Audio]]:
        """given a speaker id string, returns a list of raw wav array"""
        pass

    # @abstractmethod
    # def __ior__(self, __value: "SpeakerAudioProvider") -> None:
    #     pass


class RandomDataset(SpeakerAudioProvider):
    def __init__(self, num_speaker: int, utt_range: Tuple[int, int]) -> None:
        self.data = {
            str(i): [
                LoadedAudio(
                    Audio(wav=torch.rand(1, random.randint(*utt_range)), fs=44100)
                )
                for _ in range(random.randint(2, 10))
            ]
            for i in range(num_speaker)
        }

    def get_speaker_list(self) -> Iterable[str]:
        return self.data.keys()

    def get_speaker_files(self, name: str) -> Iterable[LazyLoadable[Audio]]:
        return self.data.get(name, [])


class TSEWavDataset(SpeakerAudioProvider):
    @classmethod
    def from_folder(cls, folder: str) -> "TSEWavDataset":
        data = {}
        for spk in glob(folder):
            if os.path.isdir(os.path.join(folder, spk)):
                data.setdefault(spk, []).extend(
                    glob(os.path.join(folder, spk, "*.wav"))
                )
        return cls(data)

    def __init__(self, data: Dict[str, List[LazyLoadable[Audio]]]) -> None:
        self.data = data

    def get_speaker_list(self) -> Iterable[str]:
        return self.data.keys()

    def get_speaker_files(self, name: str) -> Iterable[LazyLoadable[Audio]]:
        return self.data.get(name, [])


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


@dataclass
class TSEDatasetArgs:
    train_val_test_ratio = [0.8, 0.1, 0.1]


@dataclass
class TSEItem:
    mix: torch.Tensor
    ref: torch.Tensor
    y: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.mix, self.ref, self.y))

    @classmethod
    def from_file(cls, path: Path):
        with open(path, "rb") as f:
            (mix, ref, y) = pickle.load(f)
            return cls(mix, ref, y)

    def to_file(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump((self.mix, self.ref, self.y), f)


@dataclass
class LoadedTSEItem(LazyLoadable):
    item: TSEItem

    def load(self) -> TSEItem:
        return self.item


@dataclass
class LazyLoadTSEItem(LazyLoadable):
    path: str

    def load(self) -> TSEItem:
        return super().load()


class TSEDataset(Dataset):
    def __init__(self, items: List[LazyLoadable[TSEItem]]) -> None:
        super().__init__()
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TSEItem:
        return self.items[index].load()

    def to_folder(self, p: Path):
        if p.exists() and p.is_dir():
            for i, (mix, ref, y) in enumerate(self.items):
                for k, v in zip((mix, ref, y), ("mix", "ref", "y")):
                    torch.save(k, p / v / ("%d.pt" % i))
        else:
            raise ValueError("%s doesn't exist or is not a folder" % p)

    @classmethod
    def from_folder(cls, p: Path) -> "TSEDataset":
        if p.exists() and p.is_dir():
            items = []
            for mix_file, ref_file, y_file in zip(
                *(glob(str(p / v / "*.pt")) for v in ("mix", "ref", "y"))
            ):
                mix, ref, y = map(torch.load, (mix_file, ref_file, y_file))
                items.append(TSEItem(mix, ref, y))
            return cls(items)
        else:
            raise ValueError(
                "%s doesn't exist, or is not a folder, or %s/mix %s/ref %s/y doesn't all exist"
                % (p, p, p, p)
            )


def tse_item_collate_fn(
    batch: List[TSEItem]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_length = max(mixed.shape[-1] for mixed, _, __ in batch)

    mixed_wavs, ref_wavs, clean_wavs = zip(*batch)
    batch_padded_mixed_wav = pad_seq_n_stack(list(mixed_wavs), pad_length)
    batch_padded_clean_wav = pad_seq_n_stack(list(clean_wavs), pad_length)

    ref_pad_length = max(ref.shape[-1] for _, ref, __ in batch)
    batch_padded_ref_wav = pad_seq_n_stack(list(ref_wavs), ref_pad_length)

    return (batch_padded_mixed_wav, batch_padded_ref_wav, batch_padded_clean_wav)


class TSEDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=tse_item_collate_fn)


@dataclass
class TSEDatasetBuilder(Dataset):
    train: TSEDataset
    eval: TSEDataset
    test: TSEDataset

    @staticmethod
    def __split(ls: List[T], proportions: List[float]) -> List[List[T]]:
        n = len(ls)
        presum_proportions = [0.0]
        for p in proportions:
            presum_proportions.append(presum_proportions[-1] + p)
        assert presum_proportions[-1] == 1
        res = []
        for i in range(len(proportions)):
            st = int(presum_proportions[i] * n)
            en = int(presum_proportions[i + 1] * n)
            res.append(ls[st:en])
        return res

    @staticmethod
    def __gen_mix(spks: List[str], provider: SpeakerAudioProvider) -> List[TSEItem]:
        spk_utts = {spk: list(provider.get_speaker_files(spk)) for spk in spks}
        c = 0
        res: List[TSEItem] = []
        while c < 200:
            y_spk = random.choice(spks)
            if len(spk_utts[y_spk]) < 2:
                c += 1
                continue
            candidate_spk = [spk for spk in spks if spk != y_spk and spk_utts[spk]]
            if not candidate_spk:
                c += 1
                continue
            d_spk = random.choice(candidate_spk)
            # pick y, ref
            random.shuffle(spk_utts[y_spk])
            y = spk_utts[y_spk].pop().load()
            ref = spk_utts[y_spk].pop().load()
            # pick noise
            random.shuffle(spk_utts[d_spk])
            noise = spk_utts[d_spk].pop().load()
            # determine loudness of y and noise
            y_lufs = random.randint(-33, -25)
            noise_lufs = random.randint(-33, -25)
            # apply loudness
            y.wav = normalize_loudness_torch(y.wav, 48000, y_lufs)
            noise.wav = normalize_loudness_torch(noise.wav, 48000, noise_lufs)
            # get some random start point in the noise
            y_start = random.randint(-y.len, noise.len)
            noise_start = 0
            # normalize to ensure y_start and noise_start >= 0
            if y_start < 0:
                noise_start = -y_start
                y_start = 0
            end = max(y.len + y_start, noise.len + noise_start)
            y_pad = torch.zeros(1, end)
            y_pad[..., y_start : y_start + y.len] += y.wav
            mixed = torch.zeros(1, end)
            mixed[..., y_start : y_start + y.len] += y.wav
            mixed[..., noise_start : noise_start + noise.len] += noise.wav
            # TODO add more distortions and background noises
            res.append(TSEItem(mix=mixed, ref=ref.wav, y=y_pad))
        return res

    @classmethod
    def from_provider(
        cls, provider: SpeakerAudioProvider, args: TSEDatasetArgs
    ) -> "TSEDatasetBuilder":
        # - each utterance is only used once
        # - train, eval, test speakers don't intersect
        speakers = list(provider.get_speaker_list())
        train_spk, eval_spk, test_spk = cls.__split(speakers, args.train_val_test_ratio)
        train_mix = cls.__gen_mix(train_spk, provider)
        eval_mix = cls.__gen_mix(eval_spk, provider)
        test_mix = cls.__gen_mix(test_spk, provider)

        return cls(TSEDataset(train_mix), TSEDataset(eval_mix), TSEDataset(test_mix))

    def to_folder(self, p: Path) -> None:
        if p.exists() and p.is_dir():
            # train, eval, test
            p_tr = p / "tr"
            p_ev = p / "ev"
            p_ts = p / "ts"
            p_tr.mkdir(exist_ok=True)
            p_ev.mkdir(exist_ok=True)
            p_ts.mkdir(exist_ok=True)
            self.train.to_folder(p_tr)
            self.eval.to_folder(p_ev)
            self.test.to_folder(p_ts)
        else:
            raise ValueError("%s doesn't exist or is not a folder" % p)

    @classmethod
    def from_folder(cls, p: Path) -> "TSEDatasetBuilder":
        if (
            p.exists()
            and p.is_dir()
            and all(
                (p / subset).exists() and (p / subset).is_dir()
                for subset in ("tr", "ev", "ts")
            )
        ):
            train = TSEDataset.from_folder(p / "tr")
            eval = TSEDataset.from_folder(p / "ev")
            test = TSEDataset.from_folder(p / "ts")
            return cls(train, eval, test)
        else:
            raise ValueError(
                "%s doesn't exist, or is not a folder, or %s/tr %s/ev %s/ts doesn't all exist"
                % (p, p, p, p)
            )

    def __iter__(self) -> Iterator[TSEDataset]:
        return iter((self.train, self.eval, self.test))


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


@dataclass
class MelDatasetOutput:
    mel: torch.Tensor
    wav: torch.Tensor


class MelDataset(Dataset):
    def __init__(self, audios: List[LazyLoadable[Audio]], args: MelArgs):
        self.audios = [a for a in audios if 3 <= a.load().len_in_s]
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

        return MelDatasetOutput(mel=mel.squeeze(), wav=audio.squeeze(0))

    def __len__(self):
        return len(self.audios)

    @classmethod
    def from_speech_dataset(cls, dataset: SpeechDataset, args: MelArgs) -> "MelDataset":
        return cls(dataset.audios, args)


if __name__ == "__main__":
    import librosa
    import librosa.display
    from matplotlib import pyplot as plt
    import sys

    def plot_melspectrogram(wav, ax, fs=44100, title="Melspectrogram"):
        s = librosa.feature.melspectrogram(y=wav, sr=fs)
        librosa.display.specshow(
            librosa.power_to_db(s),
            x_axis="time",
            y_axis="mel",
            ax=ax,
            sr=fs,
            cmap="magma",
        )
        ax.set(title=title)

    dataset = GenshinDataset.from_parquets(sys.argv[1])
    tse_train, tse_eval, tse_test = TSEDatasetBuilder.from_provider(
        dataset, TSEDatasetArgs()
    )
    loader = TSEDataLoader(tse_train, batch_size=8, shuffle=True)
    for batch in loader:
        print(batch)
        break
    for padded_x_hat, padded_ref, padded_y_hat in loader:
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 8))

        for i in range(4):
            x_hat = padded_x_hat[i].numpy()
            y_hat = padded_y_hat[i].numpy()
            dirty = x_hat - y_hat

            ax = axes[i, 0]
            plot_melspectrogram(x_hat, ax, fs=44100, title="mixed")

            ax = axes[i, 1]
            plot_melspectrogram(y_hat, ax, fs=44100, title="clean")

            ax = axes[i, 2]
            plot_melspectrogram(dirty, ax, fs=44100, title="dirty")

            ax = axes[i, 3]
            plot_melspectrogram(padded_ref[i].numpy(), ax, fs=44100, title="ref")

        plt.tight_layout()
        plt.show()

        break
