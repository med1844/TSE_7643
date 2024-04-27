from functools import partial
from typing import Dict, Any, Optional
from pathlib import Path
from ray import tune
from ray.train import ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler
from lightning.pytorch.trainer import Trainer
import click
from tse_ln_module import TrainArgs, TSEModule
from dataset import TSEDatasetBuilder, TSEDataLoader


search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-0),
    "adamw_betas": tune.choice([(0.9, 0.999), (0.95, 0.995)]),
}


def train_fn(dataset_path: str, args: TrainArgs, wavlm_pt: str, config: Dict[str, Any]):
    args.learning_rate = config["learning_rate"]
    args.adamw_betas = config["adamw_betas"]
    args.tse_args.wavlm_pt = wavlm_pt
    hop_size = args.tse_args.stft_args.hop_size
    train_ds, val_ds, _ = TSEDatasetBuilder.from_folder(Path(dataset_path))
    train_loader = TSEDataLoader(
        hop_size, train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = TSEDataLoader(hop_size, val_ds, batch_size=args.batch_size)
    module = TSEModule(args)
    trainer = Trainer(
        max_epochs=args.epochs,
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(module, train_loader, val_loader)


def tune_tse(dataset_path: str, args: TrainArgs, wavlm_pt: str, num_samples=10):
    ray_trainer = TorchTrainer(
        partial(train_fn, dataset_path, args, wavlm_pt),
        scaling_config=ScalingConfig(use_gpu=True),
    )
    scheduler = ASHAScheduler(max_t=args.epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="eval_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


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
    "--wavlm_pt",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="pretrained_models/WavLM-Base+.pt",
)
@click.option(
    "--num_samples",
    "-n",
    type=int,
    default=10,
)
def main(config: Optional[str], dataset: str, wavlm_pt: str, num_samples: int):
    if config is None:
        args = TrainArgs.default()
    else:
        with open(config, "r", encoding="utf-8") as f:
            args = TrainArgs.model_validate_json(f.read())
    results = tune_tse(dataset, args, wavlm_pt, num_samples)
    print(results)
    print(results.get_best_result(metric="eval_loss", mode="min"))


if __name__ == "__main__":
    main()
