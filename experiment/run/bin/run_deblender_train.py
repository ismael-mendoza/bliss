#!/usr/bin/env python3


from pathlib import Path

import click
from pytorch_lightning.callbacks import EarlyStopping

from bliss.encoders.deblend import GalaxyEncoder
from experiment.run.training_functions import run_encoder_training

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--ae-model-path", required=True, type=str)
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("-b", "--batch-size", default=128)
@click.option("--lr", default=1e-4, type=float)
@click.option("-e", "--n-epochs", default=8000)
@click.option("--validate-every-n-epoch", default=20, type=int)
@click.option("--log-every-n-steps", default=10, type=float)
def main(
    seed: int,
    ae_model_path: str,
    train_file: str,
    val_file: str,
    batch_size: int,
    lr: float,
    n_epochs: int,
    validate_every_n_epoch: int,
    log_every_n_steps: int,
):

    ae_path = Path(ae_model_path)
    assert ae_path.exists()

    # setup model to train
    galaxy_encoder = GalaxyEncoder(ae_path, lr=lr)

    # early stoppin callback based on 'mean_max_residual'
    early_stopping_cb = EarlyStopping(
        "val/mean_max_residual",
        min_delta=0.1,
        patience=10,
        strict=True,
        check_on_train_epoch_end=False,
        mode="min",
    )

    run_encoder_training(
        seed=seed,
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model=galaxy_encoder,
        model_name="deblender",
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=None,
        log_every_n_steps=log_every_n_steps,
        early_stopping_cb=early_stopping_cb,
        keep_padding=True,
    )


if __name__ == "__main__":
    main()
