from typing import Optional
from pathlib import Path
from lightning.pytorch.trainer import Trainer
import click
from dataset import (
    TSEDatasetBuilder,
    TSEDataLoader,
)
from tse_ln_module import TrainArgs, TSEModule


def train(args: TrainArgs, dataset_path: str, model_ckpt: str):
    hop_size = args.tse_args.stft_args.hop_size
    *_, test_ds = TSEDatasetBuilder.from_folder(Path(dataset_path))
    test_loader = TSEDataLoader(hop_size, test_ds, batch_size=args.batch_size)
    module = TSEModule(args)
    trainer = Trainer()
    trainer.test(module, test_loader, model_ckpt)


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
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def main(config: Optional[str], dataset: str, model_ckpt: str):
    if config is None:
        args = TrainArgs.default()
    else:
        with open(config, "r", encoding="utf-8") as f:
            args = TrainArgs.model_validate_json(f.read())
    train(args, dataset, model_ckpt)


if __name__ == "__main__":
    main()
