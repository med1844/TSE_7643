from typing import Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
from torchmetrics.aggregation import MeanMetric
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.functional.audio.snr import (
    scale_invariant_signal_noise_ratio as si_snr,
)
from pydantic import BaseModel
import click
from modules.adapted_wavlm import AdaptedWavLMArgs
from modules.adapted_x_vec import AdaptedXVectorArgs
from modules.mask_predictor import MaskPredictorArgs
from modules.tse_model import TSEModelArgs, TSEModel
from dataset import (
    TSEDatasetBuilder,
    TSEDataLoader,
    TSEPredictItem,
    TSETrainItem,
)


class TrainArgs(BaseModel):
    exp_name: str
    epochs: int
    batch_size: int
    tse_args: TSEModelArgs
    learning_rate: float
    adamw_betas: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0.999

    @classmethod
    def default(cls) -> "TrainArgs":
        return cls(
            exp_name="tse",
            epochs=10,
            batch_size=16,
            tse_args=TSEModelArgs(),
            learning_rate=2e-4,
        )


def rms_loudness(signal: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(signal**2, dim=-1, keepdim=True))


def loudness_loss(
    estimated_signal: torch.Tensor, target_signal: torch.Tensor
) -> torch.Tensor:
    estimated_loudness = rms_loudness(estimated_signal)
    target_loudness = rms_loudness(target_signal)
    return (torch.abs(estimated_loudness - target_loudness)).mean()


class TSEModule(LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.model = TSEModel(args.tse_args)
        self.eval_loss_mean = MeanMetric()

    def training_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y = self.model(TSEPredictItem(mix, ref))
        loss = (
            nn.functional.l1_loss(est_y, y)
            + loudness_loss(est_y, y)
            - si_snr(est_y, y).mean()
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y = self.model(TSEPredictItem(mix, ref))
        #! use ref or y here?
        # use y since we want to know how well TSE works against ground truth
        si_snr_loss = si_snr(est_y, y).mean()
        l1_loss = nn.functional.l1_loss(est_y, y)
        loudness_loss_val = loudness_loss(est_y, y)
        total_loss = l1_loss + loudness_loss_val - si_snr_loss
        self.eval_loss_mean.update(total_loss)

    def on_validation_epoch_end(self):
        avg_loss = self.eval_loss_mean.compute()
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.eval_loss_mean.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=self.args.adamw_betas,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.args.lr_decay
        )
        return [optimizer], [scheduler]


def train(args: TrainArgs, dataset_path: str):
    hop_size = args.tse_args.stft_args.hop_size
    train_ds, val_ds, _ = TSEDatasetBuilder.from_folder(Path(dataset_path))
    train_loader = TSEDataLoader(
        hop_size, train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = TSEDataLoader(hop_size, val_ds, batch_size=args.batch_size)

    module = TSEModule(args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=3, dirpath="checkpoints/"
    )
    logger = TensorBoardLogger("tb_logs", name=args.exp_name)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    trainer.fit(module, train_loader, val_loader)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="configs/default.json",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def main(config: Optional[str], dataset: str):
    if config is None:
        args = TrainArgs.default()
    else:
        with open(config, "r", encoding="utf-8") as f:
            args = TrainArgs.model_validate_json(f.read())
    train(args, dataset)


if __name__ == "__main__":
    main()
