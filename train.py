from models import (
    ResBlock2,
    GeneratorArgs,
)
from modules import SpeechReconstructorModule
from dataset import SpeechDataset, GenshinDataset, SpeechDataLoader
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch.utils.data
from dataclasses import dataclass
from serde import serde


@serde
@dataclass
class TrainArgs:
    exp_name: str
    epochs: int
    batch_size: int
    generator_args: GeneratorArgs
    optimizer_args: OptimizerArgs
    scheduler_args: SchedulerArgs


def train(args: TrainArgs, parquets_folder: str):
    dataset = SpeechDataset.from_speaker_audio_provider(
        GenshinDataset.from_parquets(parquets_folder)
    )
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    train_loader = SpeechDataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_loader = SpeechDataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
    )

    sr_module = SpeechReconstructorModule(args.generator_args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="exp/%s/checkpoints/" % args.exp_name,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    trainer = Trainer(
        logger=TensorBoardLogger("tb_logs", name=args.exp_name),
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
    )
    trainer.fit(sr_module, train_loader, eval_loader)


if __name__ == "__main__":
    train(
        TrainArgs(
            exp_name="speech_reconstruction",
            epochs=10,
            batch_size=16,
            generator_args=GeneratorArgs(
                initial_channel=80,  # the output of WavLM, what's the structure?
                resblock=ResBlock2,
                resblock_kernel_sizes=[3, 5, 7],
                resblock_dilation_sizes=[[1, 2], [2, 6], [3, 12]],
                upsample_rates=[8, 8, 4],
                upsample_initial_channel=256,
                upsample_kernel_sizes=[16, 16, 8],
            ),
            optimizer_args=OptimizerArgs(learning_rate=0.0002),
            scheduler_args=SchedulerArgs(),
        ),
        "/mnt/d/Download/genshin_voice_3.5/genshin-voice-v3.5-mandarin/data/",
    )
