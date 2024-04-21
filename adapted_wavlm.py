import torch
import torch.nn as nn
from modules.wavlm.WavLMBasePlus import WavLMBasePlus
from modules.wavlm.WavLM import TransformerSentenceEncoderLayer
from pydantic import BaseModel
from typing import Callable


class TransformerAdaptationLayer(nn.Module):
    def __init__(self, dim_model, activation_function):
        super().__init__()
        self.transformer_layer = TransformerSentenceEncoderLayer(
            embedding_dim=dim_model,
            ffn_embedding_dim=dim_model * 4,
            num_attention_heads=8,
            dropout=0.1,
            activation_function=activation_function
        )

    def forward(self, x, spk_emb):
        return self.transformer_layer(x, spk_emb)

class AdaptedWavLMArgs(BaseModel):
      adaptation_layer_size: int = 768
      num_adaptation_layers: int = 1
      dropout_rate: float = 0.1
      activation_function: callable = nn.ReLU
      adaptation_layer_lr: float = 1e-4

class AdaptedWavLM(nn.Module):
    def __init__(self, args: AdaptedWavLMArgs, device=None):
        super().__init__()
        self.wavlm_base_plus = WavLMBasePlus(device=device)
        self.adaptation_layers = nn.ModuleList([
            TransformerAdaptationLayer(self.wavlm_base_plus.hidden_dim, args.activation_function())
            for _ in range(args.num_adaptation_layers)
        ])

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
        #   or maybe just vanilla transformer block
        # - copy & modify the forward function to add support of conditioning using spk_emb
        features = self.wavlm_base_plus.extract_features(mix)
        spk_emb = spk_emb.expand(-1, features.size(1), -1)  # Expand speaker embeddings to match temporal dimension of features
        for layer in self.adaptation_layers:
            features = layer(features, spk_emb)
        return features
