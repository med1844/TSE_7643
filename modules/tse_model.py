import torch
import torch.nn as nn
import torch.optim
from transformers import WavLMConfig
from modules.adapted_wavlm import AdaptedWavLM, AdaptedWavLMArgs
from modules.mask_predictor import MaskPredictor, MaskPredictorArgs
from dataset import TSEPredictItem
from speechbrain.pretrained import EncoderClassifier
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
    adapted_wavlm_arg: AdaptedWavLMArgs
    stft_args: STFTArgs = STFTArgs()


class TSEModel(nn.Module):
    def __init__(self, args: TSEModelArgs) -> None:
        super().__init__()
        self.args = args
        self.adapted_wavlm = AdaptedWavLM(args.adapted_wavlm_arg)
        self.x_vector = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
        )
        self.x_vector.eval()
        self.mask_predictor = MaskPredictor(
            MaskPredictorArgs(
                wavlm_dim=WavLMConfig().hidden_size, fft_dim=args.stft_args.win_size
            )
        )

    def forward(self, batch: TSEPredictItem) -> torch.Tensor:
        mix, ref = batch
        # mix: B x T, ref: B x T', y: B x T
        with torch.no_grad():
            spk_emb = self.x_vector.encode_batch(ref)
        wavlm_features = self.adapted_wavlm(
            torchaudio.functional.resample(mix, 48000, 16000), spk_emb
        )
        stft_args = self.args.stft_args
        window = stft_args.window.to(mix.device)
        mix_stft = torch.stft(
            mix,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=window,
            return_complex=True,
        )
        mix_stft = torch.einsum("bnt -> btn", mix_stft)
        mix_mag = mix_stft.abs()
        mix_phase = mix_stft.angle()
        dup_wavlm_features = wavlm_features.repeat_interleave(4, dim=1)
        padding_needed = mix_mag.shape[1] - dup_wavlm_features.shape[1]
        dup_pad_wavlm_features = nn.functional.pad(
            dup_wavlm_features, (0, 0, 0, padding_needed), mode="replicate"
        )
        mask = self.mask_predictor(dup_pad_wavlm_features, mix_mag)
        est_y_mag = mix_mag * mask
        est_y_stft = torch.einsum("btn -> bnt", torch.polar(est_y_mag, mix_phase))
        est_y = torch.istft(
            est_y_stft,
            n_fft=stft_args.win_size,
            hop_length=stft_args.hop_size,
            win_length=stft_args.win_size,
            window=window,
        )
        return est_y
