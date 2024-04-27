from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.optim
from torchaudio.backend import torchaudio
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

    def train_loss(
        self,
        est_y_spec: Spectrogram,
        y: torch.Tensor,
        adapted_wavlm_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compare adapted wavlm feature with real y feature
        with torch.no_grad():
            y_features = self.model.adapted_wavlm.wavlm_base_plus(
                torchaudio.functional.resample(y, 48000, 16000)
            )
        wavlm_loss = nn.functional.mse_loss(adapted_wavlm_features, y_features.clone())

        # calculate time domain loss
        est_y = est_y_spec.to_wav(self.args.tse_args.stft_args)
        y_loss = -si_snr(est_y, y).mean()

        return y_loss, wavlm_loss

    def __log_audio_n_spec(
        self,
        mix: torch.Tensor,
        ref: torch.Tensor,
        y: torch.Tensor,
        est_y_spec: Spectrogram,
        mask: torch.Tensor,
    ):
        self.logger.experiment.add_image(
            "mix_spec",
            Spectrogram.from_wav(self.args.tse_args.stft_args, mix).plot_to_tensor(),
            global_step=self.global_step,
        )
        y_spec = Spectrogram.from_wav(self.args.tse_args.stft_args, y)
        self.logger.experiment.add_image(
            "est_y_spec", est_y_spec.plot_to_tensor(), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "y_spec", y_spec.plot_to_tensor(), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "mask", mask.detach()[0], global_step=self.global_step, dataformats="WH"
        )
        self.logger.experiment.add_audio(
            "mix", mix[0], global_step=self.global_step, sample_rate=48000
        )
        self.logger.experiment.add_audio(
            "ref", ref[0], global_step=self.global_step, sample_rate=48000
        )
        self.logger.experiment.add_audio(
            "y", y[0], global_step=self.global_step, sample_rate=48000
        )
        self.logger.experiment.add_audio(
            "est_y",
            est_y_spec.to_wav(self.args.tse_args.stft_args).detach()[0],
            global_step=self.global_step,
            sample_rate=48000,
        )

    def __log_loss(
        self,
        y_loss: torch.Tensor,
        wavlm_loss: torch.Tensor,
        phase: Literal["train", "eval"],
    ):
        self.log(
            f"{phase}_loss",
            y_loss + wavlm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{phase}_y_si_snr_loss",
            y_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{phase}_wavlm_loss",
            wavlm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y_spec, mask, adapted_wavlm_features = self.model(TSEPredictItem(mix, ref))
        y_loss, wavlm_loss = self.train_loss(est_y_spec, y, adapted_wavlm_features)
        loss = y_loss + wavlm_loss

        if torch.isnan(loss):
            print("nan detected!")
            exit()
        self.__log_loss(y_loss, wavlm_loss, "train")
        if batch_idx % 100 == 0:
            self.__log_audio_n_spec(mix, ref, y, est_y_spec, mask)
        return loss

    def validation_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y_spec, mask, adapted_wavlm_features = self.model(TSEPredictItem(mix, ref))
        #! use ref or y here?
        # use y since we want to know how well TSE works against ground truth

        y_loss, wavlm_loss = self.train_loss(est_y_spec, y, adapted_wavlm_features)
        self.__log_loss(y_loss, wavlm_loss, "eval")
        if batch_idx % 10 == 0:
            self.__log_audio_n_spec(mix, ref, y, est_y_spec, mask)

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=self.args.adamw_betas,
        )
        return optimizer
