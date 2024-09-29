#!/usr/bin/env python3
import click

from bliss.encoders.detection import DetectionEncoder
from experiment.run.training_functions import run_encoder_training


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("--ds-seed", required=True, type=int, help="Random seed used for dataset")
@click.option("-b", "--batch-size", default=32)
@click.option("-e", "--n-epochs", default=25)
@click.option("--validate-every-n-epoch", default=1, type=int)
@click.option("--val-check-interval", default=40, type=int, help="# of training batches")
@click.option("--log-every-n-steps", default=10, type=int)
def main(
    seed: int,
    train_file: str,
    val_file: str,
    ds_seed: int,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    val_check_interval: int,
    log_every_n_steps: int,
):

    # for logging
    info = {
        "ds_seed": ds_seed,
        "train_file": train_file,
        "val_file": val_file,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "validate_every_n_epoch": validate_every_n_epoch,
        "lr": 1e-4,
    }

    model = DetectionEncoder()
    run_encoder_training(
        seed=seed,
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model=model,
        model_name="detection",
        validate_every_n_epoch=validate_every_n_epoch,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        log_info_dict=info,
    )


if __name__ == "__main__":
    main()
