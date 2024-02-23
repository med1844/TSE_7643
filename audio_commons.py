from typing import BinaryIO, Union
import torch
import torchaudio


def read_wav_at_fs(target_fs: int, filename: Union[BinaryIO, str]) -> torch.Tensor:
    y, fs = torchaudio.load(filename)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if fs != target_fs:
        y = torchaudio.functional.resample(y, fs, target_fs)
    return y
