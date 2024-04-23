from typing import Iterable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim
from modules.adapted_x_vec import AdaptedXVector, AdaptedXVectorArgs
from modules.adapted_wavlm import AdaptedWavLM, AdaptedWavLMArgs
from modules.mask_predictor import MaskPredictor, MaskPredictorArgs
from dataset import TSEPredictItem
import torchaudio
from pydantic import BaseModel


class STFTArgs(BaseModel):
    n_fft: int = 2048
    hop_size: int = 240
    win_size: int = 1200

    @property
    def window(self) -> torch.Tensor:
        return torch.hann_window(self.win_size)


class TSEModelArgs(BaseModel):
    win_size: int = 1200
    spk_emb_dim: int = 512
    num_wavlm_adapt_layers: int = 1
    x_vec_adaptor_hidden_size: int = 512
    num_mask_pred_conformer_blocks: int = 16

    @property
    def adapted_wavlm_args(self) -> AdaptedWavLMArgs:
        return AdaptedWavLMArgs(
            spk_emb_dim=self.spk_emb_dim,
            num_adaptation_layers=self.num_wavlm_adapt_layers,
        )

    @property
    def adapted_x_vec_args(self) -> AdaptedXVectorArgs:
        return AdaptedXVectorArgs(
            hidden_size=self.x_vec_adaptor_hidden_size, out_size=self.spk_emb_dim
        )

    @property
    def mask_predictor_args(self) -> MaskPredictorArgs:
        return MaskPredictorArgs(
            num_conformer_blocks=self.num_mask_pred_conformer_blocks,
            fft_dim=self.win_size,
        )

    @property
    def stft_args(self) -> STFTArgs:
        return STFTArgs(win_size=self.win_size)


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
        wav_mag = wav_stft.abs()
        wav_phase = wav_stft.angle()
        wav_spec = torch.log(wav_mag**2).clamp_min(-100)
        return cls(wav_spec, wav_phase)

    def to_wav(self, stft_args: STFTArgs) -> torch.Tensor:
        copy_spec = self.spec.clone()
        copy_spec[copy_spec <= -100] = -torch.inf
        copy_mag = torch.sqrt(torch.exp(copy_spec))
        spectrum = torch.einsum("btn -> bnt", torch.polar(copy_mag, self.phase))
        wav = torch.istft(
            spectrum,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=stft_args.window.to(copy_mag.device),
        )
        return wav

    def __iter__(self) -> Iterable[torch.Tensor]:
        return iter((self.spec, self.phase))


class TSEModel(nn.Module):
    def __init__(self, args: TSEModelArgs) -> None:
        super().__init__()
        self.args = args
        self.adapted_wavlm = AdaptedWavLM(args.adapted_wavlm_args)
        self.adapted_x_vec = AdaptedXVector(args.adapted_x_vec_args)
        self.mask_predictor = MaskPredictor(args.mask_predictor_args)

    def forward(self, batch: TSEPredictItem) -> Spectrogram:
        mix, ref = batch
        # mix: B x T, ref: B x T', y: B x T
        spk_emb = self.adapted_x_vec(ref)
        wavlm_features = self.adapted_wavlm(
            torchaudio.functional.resample(mix, 48000, 16000), spk_emb
        )
        mix_spec, mix_phase = Spectrogram.from_wav(self.args.stft_args, mix)
        dup_wavlm_features = wavlm_features.repeat_interleave(4, dim=1)
        padding_needed = mix_spec.shape[1] - dup_wavlm_features.shape[1]
        dup_pad_wavlm_features = nn.functional.pad(
            dup_wavlm_features, (0, 0, 0, padding_needed), mode="replicate"
        )
        mask = self.mask_predictor(dup_pad_wavlm_features, mix_spec)
        est_y_spec = mix_spec * mask
        return Spectrogram(est_y_spec, mix_phase)
