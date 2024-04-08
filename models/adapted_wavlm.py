from transformers import WavLMConfig, WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
import torch
import torch.nn as nn
from pydantic import BaseModel


class AdaptedWavLMArgs(BaseModel):
    pass


class AdaptedWavLM(nn.Module):
    def __init__(self, args: AdaptedWavLMArgs) -> None:
        super().__init__()
        self.wavlm = WavLMModel(WavLMConfig())
        self.wavlm.requires_grad_(False)

        # TODO: add adaptation layer initialization
        # TODO: add argument into `AdaptedWavLMConfig` if there's any

    def forward(self, mix: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix: B x L tensor
            spk_emb: B x F tensor
        Returns:
            adapted wavlm feature: B x T x D tensor
        References:
        - https://arxiv.org/abs/2211.00482
        - https://github.com/sinhat98/adapter-wavlm/blob/main/modeling.py
        """
        # TODO: add adaptation logic & spk_emb conditioning
        return self.wavlm(mix).extract_features
