from models.adapted_wavlm import AdaptedWavLMArgs
from modules import TSEModule, TSEArgs
from dataset import TSEDatasetBuilder, TSEDataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import click


class TrainArgs(BaseModel):
    exp_name: str
    epochs: int
    batch_size: int
    sr_args: TSEArgs

    @classmethod
    def default(cls) -> "TrainArgs":
        return cls(
            exp_name="speech_reconstruction",
            epochs=10,
            batch_size=16,
            sr_args=TSEArgs(
                adapted_wavlm_config=AdaptedWavLMArgs(),
            ),
        )


def train(args: TrainArgs, dataset_path: str):
    train, eval, test = TSEDatasetBuilder.from_folder(Path(dataset_path))
    train_loader = TSEDataLoader(train)
    eval_loader = TSEDataLoader(eval)
    sr_module = TSEModule(args.sr_args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="exp/%s/checkpoints/" % args.exp_name,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    trainer = Trainer(
        logger=WandbLogger(project=args.exp_name),
        # logger=TensorBoardLogger("tb_logs", name=args.exp_name),
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
    )
    trainer.fit(sr_module, train_loader, eval_loader)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="configs/v0.json",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def main(config: Optional[str], dataset: str):
    # TODO remove parquets folder from args
    if config is None:
        args = TrainArgs.default()
    else:
        with open(config, "r", encoding="utf-8") as f:
            args = TrainArgs.model_validate_json(f.read())
    train(args, dataset)


if __name__ == "__main__":
    main()
