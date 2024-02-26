from typing import List, Tuple
from audio_commons import plot_spectrogram
import lightning as ln
from models.generator import Generator, GeneratorArgs
from models.mpd import MultiPeriodDiscriminator
from models.msd import MultiScaleDiscriminator
from transformers import WavLMConfig, WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
import torch
import itertools
from dataclasses import dataclass
from losses import feature_loss, discriminator_loss, generator_loss
from dataset import MelDatasetOutput, mel_spectrogram, MelArgs
import torchaudio
from traits import SerdeJson, Json


@dataclass
class SpeechReconstructorArgs(SerdeJson):
    learning_rate: float
    generator_args: GeneratorArgs
    mel_args: MelArgs
    adam_beta: Tuple[float, float] = (0.9, 0.99)
    lr_decay: float = 0.999

    def to_json(self) -> Json:
        return {
            "learning_rate": self.learning_rate,
            "generator_args": self.generator_args.to_json(),
            "mel_args": self.mel_args.to_json(),
            "adam_beta": self.adam_beta,
            "lr_decay": self.lr_decay,
        }

    @classmethod
    def from_json(cls, obj: Json) -> "SpeechReconstructorArgs":
        return cls(
            learning_rate=obj["learning_rate"],
            generator_args=GeneratorArgs.from_json(obj["generator_args"]),
            mel_args=MelArgs.from_json(obj["mel_args"]),
            adam_beta=tuple(obj["adam_beta"]),
            lr_decay=obj["lr_decay"],
        )


class SpeechReconstructorModule(ln.LightningModule):
    def __init__(self, args: SpeechReconstructorArgs) -> None:
        super().__init__()
        self.args = args
        self.wavlm = WavLMModel(WavLMConfig())
        self.generator = Generator(args.generator_args)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.automatic_optimization = False

        self.wavlm.requires_grad_(False)

    def training_step(self, in_: MelDatasetOutput) -> None:
        # https://github.com/jik876/hifi-gan/blob/master/train.py
        optim_g, optim_d = self.optimizers(use_pl_optimizer=True)

        mel_args = self.args.mel_args
        y_mel, y = in_
        x: Wav2Vec2BaseModelOutput = self.wavlm(
            torchaudio.functional.resample(y, 48000, 16000)
        )
        y_g_hat = self.generator(
            x.extract_features.permute(0, 2, 1)  # (B, T, D) -> (B, D, T), D = 512
        )
        y_g_hat = y_g_hat[..., : y.shape[-1]]
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            mel_args.n_fft,
            mel_args.num_mels,
            mel_args.sampling_rate,
            mel_args.hop_size,
            mel_args.win_size,
            mel_args.fmin,
            mel_args.fmax,
        )

        optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y.unsqueeze(1), y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y.unsqueeze(1), y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        self.log("train_loss_d", loss_disc_all)
        self.manual_backward(loss_disc_all)
        optim_d.step()

        optim_g.zero_grad()
        # L1 Mel-Spectrogram Loss
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y.unsqueeze(1), y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y.unsqueeze(1), y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        self.log("train_loss_g", loss_gen_all)
        self.log("train_loss_gen_s", loss_gen_s)
        self.log("train_loss_gen_f", loss_gen_f)
        self.log("train_loss_fm_s", loss_fm_s)
        self.log("train_loss_fm_f", loss_fm_f)
        self.log("train_loss_mel", loss_mel)
        self.manual_backward(loss_gen_all)
        optim_d.step()

    def validation_step(self, in_: MelDatasetOutput) -> torch.Tensor:
        mel_args = self.args.mel_args
        y_mel, y = in_
        x: Wav2Vec2BaseModelOutput = self.wavlm(
            torchaudio.functional.resample(y, 48000, 16000)
        )
        y_g_hat = self.generator(x.extract_features.permute(0, 2, 1))
        y_g_hat = y_g_hat[..., : y.shape[-1]]
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            mel_args.n_fft,
            mel_args.num_mels,
            mel_args.sampling_rate,
            mel_args.hop_size,
            mel_args.win_size,
            mel_args.fmin,
            mel_args.fmax,
        )
        val_loss = torch.nn.functional.l1_loss(y_mel, y_g_hat_mel)
        self.log("val_loss", val_loss)

        self.logger.log_image("y_mel", [plot_spectrogram(y_mel.cpu().numpy()[0])])
        self.logger.log_image(
            "y_g_hat_mel",
            [plot_spectrogram(y_g_hat_mel.cpu().numpy()[0])],
        )

        return val_loss

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
