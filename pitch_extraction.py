import torch
import torchaudio
import torchcrepe
import numpy as np


def get_pitch(wav_data: torch.Tensor, mel: torch.Tensor, hop_size: int, threshold=0.3):
    # https://github.com/openvpi/DiffSinger/blob/04bc269c9d94ad4a0ea076188014d476d6902c6f/inference/val_nsf_hifigan.py#L8
    wav16k = torchaudio.functional.resample(wav_data, 44100, 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0)

    f0_min = 40
    f0_max = 800

    f0, pd = torchcrepe.predict(
        wav16k_torch,
        16000,
        80,
        f0_min,
        f0_max,
        pad=True,
        model="full",
        batch_size=1024,
        return_periodicity=True,
    )

    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(len(mel)) * hop_size / 44100
    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    return f0
