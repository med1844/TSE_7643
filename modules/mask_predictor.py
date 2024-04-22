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

    wavlm_dim: int
    fft_dim: int
    # add model specific parameters here, e.g. num_conformer_blocks


class MaskPredictor(nn.Module):
    def __init__(self, args: MaskPredictorArgs) -> None:
        super().__init__()
        real_fft_dim = (
            (args.fft_dim >> 1) + 1
        )  # 1200 -> 601, 2048 -> 1025, see https://pytorch.org/docs/stable/generated/torch.stft.html
        self.example_layer = nn.Sequential(
            nn.Linear(args.wavlm_dim + real_fft_dim, real_fft_dim),
            nn.LayerNorm(real_fft_dim),
            nn.ReLU(),
        )
        self.attention = nn.Linear(args.wavlm_dim, 1)
        self.rnn = nn.GRU(args.wavlm_dim + real_fft_dim, args.wavlm_dim, batch_first=True)
        self.fc = nn.Linear(args.wavlm_dim, real_fft_dim)


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
        # TODO: implement mask predictor, use adapted_wavlm_feature to condition the model
        # feel free to delete the example fcn, it's just to ensure the dimensions are correct
       
        # Concatenate adapted_wavlm_feature and mix_mag along the feature dimension
        concat_features = torch.cat((mix_mag, adapted_wavlm_feature), dim=-1)

        # Apply attention mechanism to compute attention weights
        attention_weights = F.softmax(self.attention(adapted_wavlm_feature), dim=1)

        # Apply attention weights to adapted_wavlm_feature
        context_vector = torch.sum(attention_weights * adapted_wavlm_feature, dim=1, keepdim=True)

        # Concatenate context_vector with mix_mag along the feature dimension
        conditioned_features = torch.cat((mix_mag, context_vector.expand(-1, mix_mag.size(1), -1)), dim=-1)

        # Pass conditioned_features through RNN
        rnn_output, _ = self.rnn(conditioned_features)

        # Apply fully connected layer to obtain predicted mask
        predicted_mask = self.fc(rnn_output)
        return predicted_mask
        # return self.example_layer(
        #     torch.concat((mix_mag, adapted_wavlm_feature), dim=-1)
        # )
