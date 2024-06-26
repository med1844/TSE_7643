from typing import Optional
import torch
import torch.nn as nn
from modules.wavlm.WavLMBasePlus import WavLMBasePlus
from modules.wavlm.WavLM import MultiheadAttention
from pydantic import BaseModel
from modules.cln import SCLN


class AdaptedTransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        spk_emb_dim: int = 512,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = "relu"
        self.activation_fn = nn.ReLU()
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = SCLN(spk_emb_dim, self.embedding_dim)

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = SCLN(spk_emb_dim, self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        spk_emb: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x, spk_emb)
            x, *_ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x, spk_emb)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, *_ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x, spk_emb)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x, spk_emb)

        return x


class AdaptedWavLMArgs(BaseModel):
    spk_emb_dim: int = 512
    num_adaptation_layers: int = 1
    wavlm_pt: str = "pretrained_models/WavLM-Base+.pt"


class AdaptedWavLM(nn.Module):
    def __init__(self, args: AdaptedWavLMArgs):
        super().__init__()
        self.wavlm_base_plus = WavLMBasePlus(args.wavlm_pt)
        self.adaptation_layers = nn.ModuleList(
            [
                AdaptedTransformerSentenceEncoderLayer(args.spk_emb_dim)
                for _ in range(args.num_adaptation_layers)
            ]
        )

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
        # model = self.wavlm_base_plus.model
        # features = model.feature_extractor(mix)
        # features = features.transpose(1, 2)
        # features = model.layer_norm(features)

        # if model.post_extract_proj is not None:
        #     features = model.post_extract_proj(features)

        # features = model.dropout_input(features)

        features = self.wavlm_base_plus(mix)
        adapt_features = features.clone()
        for layer in self.adaptation_layers:
            adapt_features = layer(adapt_features, spk_emb)
        return features + adapt_features

        # x, _ = model.encoder(features + adapt_features)
        # return x
