#!/usr/bin/env python3
import typer

from bliss.encoders.detection import DetectionEncoder
from bliss.training_functions import run_encoder_training


def main(
    seed: int = typer.Option(),
    train_file: str = typer.Option(),
    val_file: str = typer.Option(),
    batch_size: int = 32,
    n_epochs: int = 50,
    validate_every_n_epoch: int = 1,
    log_every_n_steps: int = 50,
    val_check_interval: float = 0.2,
    version: int = 0,
):
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
        version=version,
    )


if __name__ == "__main__":
    typer.run(main)
