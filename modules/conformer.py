# forked from https://github.com/Sanyuan-Chen/CSS_with_Conformer/blob/master/nnet/conformer.py

import math
from typing import Optional, Tuple
import torch
from torch import nn
import numpy
from modules.cln import SCLN


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, maxlen: int = 1000, embed_v: bool = False) -> None:
        super().__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2 * maxlen, d_model)
        if embed_v:
            self.pe_v = torch.nn.Embedding(2 * maxlen, d_model)
        self.embed_v = embed_v

    def forward(
        self, pos_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None


class AttnBiasMHA(nn.Module):
    # TODO migrate to flash attn after PR merge: https://github.com/Dao-AILab/flash-attention/pull/617
    """Multi-Head Attention layer with attention bias support.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(
        self, n_head: int, n_feat: int, d_cond: int, dropout_rate: float
    ) -> None:
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(n_feat)
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.linear_cond = nn.Linear(d_cond, n_feat)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_k: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(self.linear_cond(cond)).view(
            n_batch, -1, self.h, self.d_k
        )  # (b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)  # (b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        A = torch.einsum("bthd,bshd->bhts", q, k)
        if pos_k is not None:
            reshape_q = (
                q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0, 1)
            )
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.einsum("bhts,bshc->bthc", p_attn, v).reshape(
            n_batch, -1, self.h * self.d_k
        )  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))  # (batch, time1, d_model)


class ConvModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernel_size: int,
        dropout_rate: float,
        causal: bool = False,
    ) -> None:
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.pw_conv_1 = nn.Conv2d(1, 2, 1, 1, 0)
        self.glu_act = torch.nn.Sigmoid()
        self.causal = causal
        if causal:
            self.dw_conv_1d = nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size,
                1,
                padding=(kernel_size - 1),
                groups=input_dim,
            )
        else:
            self.dw_conv_1d = nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
                groups=input_dim,
            )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.act = nn.ReLU()
        self.pw_conv_2 = nn.Conv2d(1, 1, 1, 1, 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = x[:, 0] * self.glu_act(x[:, 1])
        x = x.permute([0, 2, 1])
        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, : -(self.kernel_size - 1)]
        # change from batchnorm to layernorm due to schedulefree caveat
        x = x.permute([0, 2, 1])
        x = self.layer_norm(x)
        x = x.permute([0, 2, 1])
        x = self.act(x)
        x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.pw_conv_2(x)
        x = self.dropout(x).squeeze(1)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_inner: int, dropout_rate: float) -> None:
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        out = self.net(x)

        return out


class ConformerBlock(nn.Module):
    """
    :param int d_model: attention vector size
    :param int n_head: number of heads
    :param int d_ffn: feedforward size
    :param int kernel_size: cnn kernal size, it must be an odd
    :param int dropout_rate: dropout_rate
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffn: int,
        d_cond: int,
        kernel_size: int,
        dropout_rate: float,
        causal: bool = False,
    ) -> None:
        """Construct an EncoderLayer object."""
        super().__init__()
        self.feed_forward_in = FeedForward(d_model, d_ffn, dropout_rate)
        self.self_attn = AttnBiasMHA(n_head, d_model, d_cond, dropout_rate)
        self.conv = ConvModule(d_model, kernel_size, dropout_rate, causal=causal)
        self.feed_forward_out = FeedForward(d_model, d_ffn, dropout_rate)
        self.layer_norm = SCLN(d_cond, d_model)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        pos_k: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor c: conditional features (batch, max_time_in, c_dim)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = x + 0.5 * self.feed_forward_in(x)
        x = x + self.self_attn(x, c, pos_k, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x, c)

        return out


class ConformerEncoder(nn.Module):
    """Conformer Encoder https://arxiv.org/abs/2005.08100"""

    def __init__(
        self,
        idim: int,
        cond_dim: int,
        attention_dim=256,
        attention_heads=4,
        linear_units=1024,
        num_blocks=16,
        kernel_size=33,
        dropout_rate=0.1,
        causal=False,
        relative_pos_emb=True,
    ):
        super(ConformerEncoder, self).__init__()

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim),
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        )

        if relative_pos_emb:
            self.pos_emb = RelativePositionalEncoding(
                attention_dim // attention_heads, 1000, False
            )
        else:
            self.pos_emb = None

        self.encoders = torch.nn.Sequential(
            *[
                ConformerBlock(
                    attention_dim,
                    attention_heads,
                    linear_units,
                    cond_dim,
                    kernel_size,
                    dropout_rate,
                    causal=causal,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, xs: torch.Tensor, c: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        """
        Args:
            xs: B x T x idim tensor
            c: B x T x cond_ tensor
        Returns:
            condition layer normalized x: B x T x hidden_size tensor
        """

        xs = self.embed(xs)

        if self.pos_emb is not None:
            x_len = xs.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(xs.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, _ = self.pos_emb(pos_seq)
        else:
            pos_k = None
        for layer in self.encoders:
            xs = layer(xs, c, pos_k, masks)

        return xs
