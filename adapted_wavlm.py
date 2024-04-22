from typing import Optional
import torch
import torch.nn as nn
from modules.wavlm.WavLMBasePlus import WavLMBasePlus
from modules.wavlm.WavLM import MultiheadAttention
from pydantic import BaseModel


# https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer/blob/main/model/blocks.py#L8
class SCLN(nn.Module):
    """Speaker Condition Layer Normalization"""

    def __init__(self, s_size: int, hidden_size: int, eps=1e-8, bias=False):
        super(SCLN, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = nn.Linear(
            s_size,
            2 * hidden_size,  # For both b (bias) and g (gain)
            bias,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        Args:
            x: B x T x hidden_size tensor
            spk_emb: B x s_size tensor
        Returns:
            condition layer normalized x: B x T x hidden_size tensor
        """
        mu, sigma = (
            torch.mean(x, dim=-1, keepdim=True),
            torch.std(x, dim=-1, keepdim=True),
        )
        y = (x - mu) / (sigma + self.eps)  # [B, T, hidden_size]

        # [B, 1, 2 * hidden_size] --> 2 * [B, 1, hidden_size]
        b, g = torch.split(self.affine_layer(s), self.hidden_size, dim=-1)

        o = g * y + b  # [B, T, hidden_size]

        return o


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


class AdaptedWavLM(nn.Module):
    def __init__(self, args: AdaptedWavLMArgs, device=None):
        super().__init__()
        self.wavlm_base_plus = WavLMBasePlus(device=device)
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
        model = self.wavlm_base_plus.model
        features = model.feature_extractor(mix)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if model.post_extract_proj is not None:
            features = model.post_extract_proj(features)

        features = model.dropout_input(features)

        # put adaptation layer here
        for layer in self.adaptation_layers:
            features = layer(features, spk_emb)

        x, _ = model.encoder(features)
        return x
