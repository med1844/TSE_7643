from typing import List, Tuple
from audio_commons import plot_spectrogram
import lightning as ln
from models.generator import Generator, GeneratorArgs
from transformers import WavLMConfig, WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
import torch
import torch.nn as nn
import itertools
from dataclasses import dataclass
from losses import feature_loss, discriminator_loss, generator_loss
from dataset import MelDatasetOutput, mel_spectrogram, MelArgs
import torchaudio
from traits import SerdeJson, Json
from models.conformer import ConformerBlock, ConformerEncoder


@dataclass
class TSEArgs(SerdeJson):
    learning_rate: float
    mel_args: MelArgs
    num_conformer_blocks: int
    adam_beta: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0.999

    def to_json(self) -> Json:
        return {
            "learning_rate": self.learning_rate,
            "mel_args": self.mel_args.to_json(),
            "num_conformer_blocks": self.num_conformer_blocks,
            "adam_beta": list(self.adam_beta),
            "lr_decay": self.lr_decay,
        }

    @classmethod
    def from_json(cls, obj: Json) -> "TSEArgs":
        return cls(
            learning_rate=obj["learning_rate"],
            mel_args=MelArgs.from_json(obj["mel_args"]),
            num_conformer_blocks=obj["num_conformer_blocks"],
            adam_beta=tuple(obj["adam_beta"]),
            lr_decay=obj["lr_decay"],
        )


class TSEModule(ln.LightningModule):
    def __init__(self, args: TSEArgs) -> None:
        super().__init__()
        self.args = args
        wavlm_config = WavLMConfig()
        self.wavlm = WavLMModel(wavlm_config)
        self.generator = Generator(GeneratorArgs.default())  # we use pretrained model
        self.conformers = ConformerEncoder(wavlm_config.hidden_size)

        self.wavlm.requires_grad_(False)
        self.generator.requires_grad_(False)

    def training_step(self, in_: MelDatasetOutput) -> None:
        wavlm_features = self.wavlm(in_.wav)

    def validation_step(self, in_: MelDatasetOutput) -> torch.Tensor:
        pass
