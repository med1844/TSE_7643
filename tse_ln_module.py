from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.optim
from torchaudio.backend import torchaudio
from lightning.pytorch import LightningModule
from pydantic import BaseModel
from schedulefree import AdamWScheduleFree
from modules.tse_model import TSEModelArgs, TSEModel
from dataset import (
    TSEPredictItem,
    TSETrainItem,
)
from audio_commons import Spectrogram, plot_spectrogram_mask


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

    def __calc_ideal_mask(self, mix: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mix_spec = Spectrogram.from_wav(self.args.tse_args.stft_args, mix)
        y_spec = Spectrogram.from_wav(self.args.tse_args.stft_args, y)
        ideal_mask = mix_spec.spec - y_spec.spec
        return ideal_mask

    def train_loss(
        self,
        ideal_mask: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        adapted_wavlm_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compare adapted wavlm feature with real y feature
        with torch.no_grad():
            y_features = self.model.adapted_wavlm.wavlm_base_plus(
                torchaudio.functional.resample(y, 48000, 16000)
            )
        wavlm_loss = (
            nn.functional.mse_loss(adapted_wavlm_features, y_features.clone()) * 100
        )  # empirical factor

        # calculate mask loss
        mask_loss = nn.functional.mse_loss(ideal_mask, mask)

        return mask_loss, wavlm_loss

    def __log_audio_n_spec(
        self,
        mix: torch.Tensor,
        ref: torch.Tensor,
        y: torch.Tensor,
        est_y_spec: Spectrogram,
        mask: torch.Tensor,
        ideal_mask: torch.Tensor,
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
            "mask",
            plot_spectrogram_mask(mask.detach()[0].T),
            global_step=self.global_step,
        )
        self.logger.experiment.add_image(
            "ideal_mask",
            plot_spectrogram_mask(ideal_mask[0].T),
            global_step=self.global_step,
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
        mask_loss: torch.Tensor,
        wavlm_loss: torch.Tensor,
        phase: Literal["train", "eval"],
    ):
        self.log(
            f"{phase}_loss",
            mask_loss + wavlm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{phase}_mask_loss",
            mask_loss,
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

        ideal_mask = self.__calc_ideal_mask(mix, y)
        mask_loss, wavlm_loss = self.train_loss(
            ideal_mask, mask, y, adapted_wavlm_features
        )
        loss = mask_loss + wavlm_loss

        if torch.isnan(loss):
            print("nan detected!")
            exit()
        self.__log_loss(mask_loss, wavlm_loss, "train")
        if batch_idx % 100 == 0:
            self.__log_audio_n_spec(mix, ref, y, est_y_spec, mask, ideal_mask)
        return loss

    def validation_step(self, batch: TSETrainItem, batch_idx: int):
        mix, ref, y = batch
        est_y_spec, mask, adapted_wavlm_features = self.model(TSEPredictItem(mix, ref))
        #! use ref or y here?
        # use y since we want to know how well TSE works against ground truth

        ideal_mask = self.__calc_ideal_mask(mix, y)
        mask_loss, wavlm_loss = self.train_loss(
            ideal_mask, mask, y, adapted_wavlm_features
        )
        self.__log_loss(mask_loss, wavlm_loss, "eval")
        if batch_idx % 10 == 0:
            self.__log_audio_n_spec(mix, ref, y, est_y_spec, mask, ideal_mask)

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(
            self.parameters(),
            lr=self.args.learning_rate,
            betas=self.args.adamw_betas,
        )
        return optimizer
