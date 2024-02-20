from typing import List, Tuple
import lightning as ln
from models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    GeneratorArgs,
)
from transformers import WavLMConfig, WavLMModel
import torch
import itertools
from dataclasses import dataclass
from serde import serde


@serde
@dataclass
class SpeechReconstructorArgs:
    learning_rate: float
    generator_args: GeneratorArgs
    adam_beta = (0.9, 0.99)
    lr_decay = 0.999


class SpeechReconstructorModule(ln.LightningModule):
    def __init__(self, args: SpeechReconstructorArgs) -> None:
        super().__init__()
        self.args = args
        self.wavlm = WavLMModel(WavLMConfig())
        self.generator = Generator(args.generator_args)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.wavlm(x)
        return y

    def validation_step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.wavlm(x)
        return y

    def configure_optimizers(
        self
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        # for the mysterious return type, please refer to this:
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.args.learning_rate,
            betas=self.args.adam_beta,
        )
        optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr=self.args.learning_rate,
            betas=self.args.adam_beta,
        )
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.args.lr_decay
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.args.lr_decay
        )
        return [optim_g, optim_d], [scheduler_g, scheduler_d]
