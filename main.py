from models.conformer import ConformerBlockArgs
from modules import TSEModule, TSEArgs
from models.generator import GeneratorArgs
from dataset import (
    MelArgs,
    SpeechDataset,
    RandomDataset,
    MelDataset,
)
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch.utils.data
from dataclasses import dataclass
from traits import SerdeJson, Json
from typing import Optional
import json
import click


@dataclass
class TrainArgs(SerdeJson):
    exp_name: str
    epochs: int
    batch_size: int
    sr_args: TSEArgs

    def to_json(self) -> Json:
        return {
            "exp_name": self.exp_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "sr_args": self.sr_args.to_json(),
        }

    @classmethod
    def from_json(cls, obj: Json) -> "TrainArgs":
        return cls(
            exp_name=obj["exp_name"],
            epochs=obj["epochs"],
            batch_size=obj["batch_size"],
            sr_args=TSEArgs.from_json(obj["sr_args"]),
        )

    @classmethod
    def default(cls) -> "TrainArgs":
        return cls(
            exp_name="speech_reconstruction",
            epochs=10,
            batch_size=16,
            sr_args=TSEArgs(
                learning_rate=0.0002,
                mel_args=MelArgs(
                    segment_size=32768,
                    n_fft=2048,
                    num_mels=128,
                    hop_size=512,
                    win_size=2048,
                    sampling_rate=44100,
                    fmin=40,
                    fmax=16000,
                ),
                conformer_block_args=ConformerBlockArgs(
                    d_model=256, n_head=4, d_ffn=1024, kernel_size=33, dropout_rate=0.1
                ),
                num_conformer_blocks=16,
            ),
        )


def train(args: TrainArgs, parquets_folder: str):
    speech_dataset = SpeechDataset.from_speaker_audio_provider(
        # GenshinDataset.from_parquets(parquets_folder)
        RandomDataset(100, (3 * 48000, 6 * 48000))
    )
    print(speech_dataset.get_summary())

    dataset = MelDataset.from_speech_dataset(
        speech_dataset,
        args.sr_args.mel_args,
    )
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
    )

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
    "--parquets_folder",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="data/parquets",
)
def main(config: Optional[str], parquets_folder: str):
    # TODO remove parquets folder from args
    if config is None:
        args = TrainArgs.default()
    else:
        with open(config, "r", encoding="utf-8") as f:
            args = TrainArgs.from_json(json.load(f))
    train(args, parquets_folder)


if __name__ == "__main__":
    main()
