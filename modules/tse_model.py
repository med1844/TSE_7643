import torch.nn as nn
from modules.adapted_x_vec import AdaptedXVector, AdaptedXVectorArgs
from modules.adapted_wavlm import AdaptedWavLM, AdaptedWavLMArgs
from modules.mask_predictor import MaskPredictor, MaskPredictorArgs
from dataset import TSEPredictItem
import torchaudio
from pydantic import BaseModel
from audio_commons import STFTArgs, Spectrogram


class TSEModelArgs(BaseModel):
    win_size: int = 1200
    spk_emb_dim: int = 512
    num_wavlm_adapt_layers: int = 1
    x_vec_adaptor_hidden_size: int = 512
    num_mask_pred_conformer_blocks: int = 16
    wavlm_pt: str = "pretrained_models/WavLM-Base+.pt"

    @property
    def adapted_wavlm_args(self) -> AdaptedWavLMArgs:
        return AdaptedWavLMArgs(
            spk_emb_dim=self.spk_emb_dim,
            num_adaptation_layers=self.num_wavlm_adapt_layers,
            wavlm_pt=self.wavlm_pt,
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
