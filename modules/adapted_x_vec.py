import torch
import torch.nn as nn
from modules.wavlm.modules import Swish
from speechbrain.pretrained import EncoderClassifier
from pydantic import BaseModel


class AdaptedXVectorArgs(BaseModel):
    hidden_size: int = 512
    out_size: int = 512


class AdaptedXVector(nn.Module):
    def __init__(self, args: AdaptedXVectorArgs) -> None:
        super().__init__()
        self.x_vector = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        self.x_vector.eval()
        self.adaptor = nn.ModuleList(
            [
                nn.Linear(512, args.hidden_size),
                Swish(),
                nn.Linear(args.hidden_size, args.out_size),
            ]
        )

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref: B x R tensor
        Returns:
            adapted speaker embedding: B x 1 x F tensor where F = args.out_size
        """
        with torch.no_grad():
            spk_emb = self.x_vector.encode_batch(ref)
        for module in self.adaptor:
            spk_emb = module(spk_emb)
        return spk_emb
