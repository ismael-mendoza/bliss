#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping

from bliss.datasets.saved_datasets import SavedIndividualGalaxies
from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from experiment.run.training_functions import setup_training_objects

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--train-file", required=True, type=str)
@click.option("--val-file", required=True, type=str)
@click.option("-b", "--batch-size", default=128)
@click.option("-e", "--n-epochs", default=4000)
@click.option("--validate-every-n-epoch", default=10, type=int)
@click.option("--lr", default=1e-5, type=float)
def main(
    seed: int,
    train_file: str,
    val_file: str,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    lr: float,
):

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training autoencoder script...
        With seed {seed} at {now}
        validate_every_n_epoch {validate_every_n_epoch},
        batch_size {batch_size}, n_epochs {n_epochs}
        learning rate {lr}

        Using datasets: {train_file}, {val_file}
        """
        print(log_msg, file=f)

    L.seed_everything(seed)

    assert Path(train_file).exists(), f"Training dataset {train_file} is not available"
    assert Path(val_file).exists(), f"Training dataset {val_file} is not available"

    # early stoppin callback based on 'mean_max_residual'
    early_stopping_cb = EarlyStopping(
        "val/mean_max_residual",
        min_delta=0.1,
        patience=10,
        strict=True,
        stopping_threshold=3.70,
        check_on_train_epoch_end=False,
        mode="min",
    )

    # setup model to train
    autoencoder = OneCenteredGalaxyAE(lr=lr)

    with open("log.txt", "a") as g:
        train_ds = SavedIndividualGalaxies(train_file)
        val_ds = SavedIndividualGalaxies(val_file)
        train_dl, val_dl, trainer = setup_training_objects(
            train_ds,
            val_ds,
            batch_size,
            NUM_WORKERS,
            n_epochs,
            validate_every_n_epoch=validate_every_n_epoch,
            val_check_interval=None,
            model_name="autoencoder",
            log_every_n_steps=train_ds.epoch_size // batch_size,  # = number of batches in 1 epoch
            log_file=g,
            early_stopping_cb=early_stopping_cb,
        )

    trainer.fit(model=autoencoder, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
