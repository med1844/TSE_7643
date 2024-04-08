import torch
import torch.nn as nn
from conformer import ConformerBlock, ConformerEncoder
from pydantic import BaseModel


class MaskPredictorArgs(BaseModel):
    """
    Args:
        wavlm_dim: the adapted wavlm feature would have shape B x T x wavlm_dim
        fft_dim: the fft would have shape B x T x fft_dim
    """

    wavlm_dim: int
    fft_dim: int
    # add model specific parameters here, e.g. num_conformer_blocks


class MaskPredictor(nn.Module):
    def __init__(self, args: MaskPredictorArgs) -> None:
        super().__init__()

    def forward(self, adapted_wavlm_feature: torch.Tensor, mix_mag: torch.Tensor):
        """
        Args:
            adapted_wavlm_feature: B x T x D
            mix_mag: B x T x F, mixed audio magnitude
        Returns:
            predicted mask, B x T x F
        References:
        - https://arxiv.org/abs/2211.09988 for feature/spectrogram concatenation
        - https://arxiv.org/abs/2211.00482 for conditioning
        """
        # TODO: implement mask predictor, use adapted_wavlm_feature to condition the model
        return torch.ones_like(mix_mag)
