import torch
import torch.nn as nn


# https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer/blob/main/model/blocks.py#L8
class SCLN(nn.Module):
    """Speaker Condition Layer Normalization"""

    def __init__(self, s_size: int, hidden_size: int, eps=1e-8, bias=False):
        super().__init__()
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
            spk_emb: B x 1 x s_size or B x T x s_size tensor
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
