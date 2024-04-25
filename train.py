from typing import Optional
from pathlib import Path
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import click
from dataset import (
    TSEDatasetBuilder,
    TSEDataLoader,
)
from tse_ln_module import TrainArgs, TSEModule


def train(args: TrainArgs, dataset_path: str, model_ckpt: Optional[str]):
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
    trainer.fit(module, train_loader, val_loader, ckpt_path=model_ckpt)


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
@click.option(
    "--model_ckpt",
    "-m",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
def main(config: Optional[str], dataset: str, model_ckpt: Optional[str]):
    if config is None:
        args = TrainArgs.default()
    else:
        with open(config, "r", encoding="utf-8") as f:
            args = TrainArgs.model_validate_json(f.read())
    train(args, dataset, model_ckpt)


if __name__ == "__main__":
    main()
