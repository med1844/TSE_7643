from typing import Literal, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
from torchmetrics.aggregation import MeanMetric
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel
import click
from modules.adapted_wavlm import AdaptedWavLMArgs
from modules.tse_model import TSEModelArgs, TSEModel
from dataset import TSEDatasetBuilder, TSEDataLoader, TSEPredictItem, TSETrainItem


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
            tse_args=TSEModelArgs(
                adapted_wavlm_arg=AdaptedWavLMArgs(),
            ),
            learning_rate=2e-4,
        )


class TSEModule(LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.model = TSEModel(args.tse_args)
        self.eval_loss_mean = MeanMetric()

    def training_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y = self.model(TSEPredictItem(mix, ref))
        loss = nn.functional.mse_loss(est_y, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y = self.model(TSEPredictItem(mix, ref))
        loss = nn.functional.mse_loss(est_y, y)
        self.eval_loss_mean.update(loss)

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

    model = TSEModule(args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=3, dirpath="checkpoints/"
    )
    logger = TensorBoardLogger("tb_logs", name=args.exp_name)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)


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
