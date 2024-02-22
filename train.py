from modules import SpeechReconstructorModule, SpeechReconstructorArgs
from models.generator import GeneratorArgs, ResBlock2
from dataset import (
    MelArgs,
    SpeechDataset,
    GenshinDataset,
    MelDataset,
)
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
    sr_args: SpeechReconstructorArgs


def train(args: TrainArgs, parquets_folder: str):
    dataset = MelDataset.from_speech_dataset(
        SpeechDataset.from_speaker_audio_provider(
            GenshinDataset.from_parquets(parquets_folder)
        ),
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

    sr_module = SpeechReconstructorModule(args.sr_args)
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
    import sys

    train(
        TrainArgs(
            exp_name="speech_reconstruction",
            epochs=10,
            batch_size=16,
            sr_args=SpeechReconstructorArgs(
                learning_rate=0.0002,
                generator_args=GeneratorArgs(
                    # https://github.com/jik876/hifi-gan/blob/master/config_v3.json
                    # https://github.com/fishaudio/Bert-VITS2/blob/master/configs/config.json#L938
                    initial_channel=512,  # the extracted feature dim of WavLM
                    resblock=ResBlock2,
                    resblock_kernel_sizes=[3, 5, 7],
                    resblock_dilation_sizes=[[1, 2], [2, 6], [3, 12]],
                    upsample_rates=[8, 8, 4, 2, 2],
                    upsample_initial_channel=512,
                    upsample_kernel_sizes=[16, 16, 8, 2, 2],
                ),
                mel_args=MelArgs(
                    segment_size=16384,
                    n_fft=1024,
                    num_mels=80,
                    hop_size=256,
                    win_size=1024,
                    sampling_rate=48000,
                    fmin=0,
                ),
            ),
        ),
        sys.argv[1],
    )
