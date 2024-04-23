import torch
import torch.nn as nn
from modules.conformer import ConformerBlock, ConformerEncoder
from pydantic import BaseModel
import torch.nn.functional as F


class MaskPredictorArgs(BaseModel):
    """
    Args:
        wavlm_dim: the adapted wavlm feature would have shape B x T x wavlm_dim
        fft_dim: the fft would have shape B x T x fft_dim
    """

    num_conformer_blocks: int = 16
    wavlm_dim: int = 768
    fft_dim: int = 2048

    @property
    def real_fft_dim(self) -> int:
        return (
            (self.fft_dim >> 1) + 1
        )  # 1200 -> 601, 2048 -> 1025, see https://pytorch.org/docs/stable/generated/torch.stft.html


class MaskPredictor(nn.Module):
    def __init__(self, args: MaskPredictorArgs) -> None:
        super().__init__()
        self.conformer = ConformerEncoder(
            args.real_fft_dim, args.wavlm_dim, num_blocks=args.num_conformer_blocks
        )

    def forward(self, adapted_wavlm_feature: torch.Tensor, mix_mag: torch.Tensor):
        """
        Args:
            adapted_wavlm_feature: B x T x D
            mix_mag: B x T x F, mixed audio spectrogram magnitude; F = (args.fft_dim // 2) + 1
        Returns:
            predicted mask, B x T x F
        References:
        - https://arxiv.org/abs/2211.09988 for feature/spectrogram concatenation
        - https://arxiv.org/abs/2211.00482 for conditioning
        """
        return self.conformer(mix_mag, adapted_wavlm_feature)
