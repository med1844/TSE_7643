import torch
import torch.nn as nn
from modules.wavlm.WavLMBasePlus import WavLMBasePlus
from modules.wavlm.WavLM import TransformerSentenceEncoderLayer
from pydantic import BaseModel


class AdaptedWavLMArgs(BaseModel):
    pass


class AdaptedWavLM(nn.Module):
    def __init__(self, args: AdaptedWavLMArgs, device=None) -> None:
        super().__init__()
        self.wavlm = WavLMBasePlus(device=device)
        self.example_fcn = nn.Linear(512, self.wavlm.hidden_dim)  # x-vector hidden size

        # TODO: add adaptation layer initialization
        # TODO: add argument into `AdaptedWavLMConfig` if there's any

    def forward(self, mix: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix: B x L tensor
            spk_emb: B x 1 x F tensor
        Returns:
            adapted wavlm feature: B x T x D tensor
        References:
        - https://arxiv.org/abs/2211.00482
        - https://github.com/sinhat98/adapter-wavlm/blob/main/modeling.py
        """
        # TODO: add adaptation logic & spk_emb conditioning
        # 0. adaptation logic.
        # - the source code of WavLM is avaiable in modules.wavlm.WavLM.WavLM
        # - you will have to manually copy & modify the `extract_features` method logic here to get features,
        #   DON'T modify any file in modules.wavlm
        # 1. spk_emb conditioning. try to implement CLN in 2211.00482, steps:
        # - you will need a transformer block, so build a new class that wraps TransformerSentenceEncoderLayer
        # - copy & modify the forward function to add support of conditioning using spk_emb
        return self.example_fcn(spk_emb) + self.wavlm(mix)
