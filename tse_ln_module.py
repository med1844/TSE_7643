from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.optim
from torchmetrics.aggregation import MeanMetric
from lightning.pytorch import LightningModule
from torchmetrics.functional.audio.snr import (
    scale_invariant_signal_noise_ratio as si_snr,
)
from pydantic import BaseModel
from schedulefree import AdamWScheduleFree
from modules.tse_model import TSEModelArgs, TSEModel
from dataset import (
    TSEPredictItem,
    TSETrainItem,
)
from audio_commons import Spectrogram


class TrainArgs(BaseModel):
    exp_name: str
    epochs: int
    batch_size: int
    tse_args: TSEModelArgs
    learning_rate: float
    adamw_betas: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0.999
    train_loss_fn: Literal["l1", "l2"] = "l1"

    @classmethod
    def default(cls) -> "TrainArgs":
        return cls(
            exp_name="tse",
            epochs=10,
            batch_size=16,
            tse_args=TSEModelArgs(),
            learning_rate=2e-4,
        )


class TSEModule(LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.model = TSEModel(args.tse_args)
        self.train_loss_fn = (nn.MSELoss if args.train_loss_fn == "l2" else nn.L1Loss)()
        self.eval_loss_mean = MeanMetric()

    def training_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y_spec = self.model(TSEPredictItem(mix, ref))
        y_spec = Spectrogram.from_wav(self.args.tse_args.stft_args, y)
        loss = self.train_loss_fn(est_y_spec.spec, y_spec.spec)
        if torch.isnan(loss):
            print("nan detected!")
            exit()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y_spec = self.model(TSEPredictItem(mix, ref))
        #! use ref or y here?
        # use y since we want to know how well TSE works against ground truth
        est_y = est_y_spec.to_wav(self.args.tse_args.stft_args)
        loss = -si_snr(est_y, y).mean()
        self.eval_loss_mean.update(loss)

    def on_validation_epoch_end(self):
        avg_loss = self.eval_loss_mean.compute()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.eval_loss_mean.reset()

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=self.args.adamw_betas,
        )
        return optimizer