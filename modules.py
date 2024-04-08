from typing import Tuple
import torch
import torch.nn as nn
import lightning as ln
from transformers import WavLMConfig
from models.adapted_wavlm import AdaptedWavLM, AdaptedWavLMArgs
from models.mask_predictor import MaskPredictor, MaskPredictorArgs
from dataset import TSEItem
from speechbrain.pretrained import EncoderClassifier
import torchaudio
from pydantic import BaseModel


class STFTArgs(BaseModel):
    hop_size: int = 240
    win_size: int = 1200

    @property
    def window(self) -> torch.Tensor:
        return torch.hann_window(self.win_size)


class TSEArgs(BaseModel):
    adapted_wavlm_config: AdaptedWavLMArgs
    stft_args: STFTArgs = STFTArgs()
    adam_beta: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0.999


class TSEModule(ln.LightningModule):
    def __init__(self, args: TSEArgs) -> None:
        super().__init__()
        self.args = args
        self.adapted_wavlm = AdaptedWavLM(args.adapted_wavlm_config)
        self.x_vector = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
        )
        self.mask_predictor = MaskPredictor(
            MaskPredictorArgs(
                wavlm_dim=WavLMConfig().hidden_size, fft_dim=args.stft_args.win_size
            )
        )

    def training_step(self, in_: TSEItem) -> torch.Tensor:
        mix, ref, y = in_
        # mix: B x T, ref: B x T', y: B x T
        spk_emb = self.x_vector.encode_batch(ref)
        wavlm_features = self.adapted_wavlm(
            torchaudio.functional.resample(mix, 48000, 16000), spk_emb
        )
        stft_args = self.args.stft_args
        mix_stft = torch.stft(
            mix,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=stft_args.window,
            return_complex=True,
        )
        mix_mag = mix_stft.abs()
        mix_phase = mix_stft.angle()
        dup_wavlm_features = wavlm_features.repeat_interleave(
            mix_mag.shape[1] // wavlm_features.shape[1], dim=1
        )
        mask = self.mask_predictor(dup_wavlm_features, mix_mag)
        est_y_mag = mix_mag * mask
        est_y_stft = torch.polar(est_y_mag, mix_phase)
        est_y = torch.istft(
            est_y_stft,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=stft_args.window,
        )
        loss = nn.functional.mse_loss(est_y, y)
        return loss
