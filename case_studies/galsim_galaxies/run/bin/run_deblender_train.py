#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L

from bliss.datasets.galsim_blends import SavedGalsimBlends
from bliss.encoders.deblend import GalaxyEncoder
from case_studies.galsim_galaxies.run.training_functions import (
    create_dataset,
    setup_training_objects,
)

NUM_WORKERS = 0
AE_STATE_DICT = "../models/autoencoder.pt"
assert Path(AE_STATE_DICT).exists()


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-n", "--n-samples", default=1028 * 20, type=int)
@click.option("--split", default=1028 * 15, type=int)
@click.option("-b", "--batch-size", default=256)
@click.option("-e", "--n-epochs", default=10001)
@click.option("--validate-every-n-epoch", default=10, type=int)
@click.option("-o", "--overwrite", is_flag=True, default=False)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
@click.option("--only-bright", is_flag=True, default=False)
@click.option("--no-padding-galaxies", is_flag=True, default=False)
@click.option("--galaxy-density", default=185, type=float)
@click.option("--star-density", default=10, type=float)
def main(
    seed: int,
    n_samples: int,
    split: int,
    batch_size: int,
    n_epochs: int,
    validate_every_n_epoch: int,
    overwrite: bool,
    tag: str,
    only_bright: bool,
    no_padding_galaxies: bool,
    galaxy_density: float,
    star_density: float,
):

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training deblender script...
        With tag {tag} and seed {seed} at {now}
        Galaxy density {galaxy_density}, star_density {star_density}, and
        Only bright '{only_bright}', no padding galaxies '{no_padding_galaxies}'.
        n_samples {n_samples}, split {split}, validate_every_n_epoch {validate_every_n_epoch},
        val_check_interval 'None', batch_size {batch_size}, n_epochs {n_epochs}
        """
        print(log_msg, file=f)

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/data_nfs/ismael/datasets/train_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/data_nfs/ismael/datasets/val_ds_{tag}.pt"

    if overwrite:
        with open("log.txt", "a") as f:
            # for max_n_sources choice, see:
            # https://www.wolframalpha.com/input?i=Poisson+distribution+with+mean+3.5
            create_dataset(
                catsim_file="../../../data/OneDegSq.fits",
                stars_mag_file="../../../data/stars_med_june2018.fits",
                n_samples=n_samples,
                train_val_split=split,
                train_ds_file=train_ds_file,
                val_ds_file=val_ds_file,
                max_n_sources=10,
                max_shift=0.5,  # uniformly random within central slen square.
                only_bright=only_bright,
                add_galaxies_in_padding=not no_padding_galaxies,
                galaxy_density=galaxy_density,
                star_density=star_density,
                log_file=f,
            )

    if not overwrite and not Path(val_ds_file).exists():
        raise IOError("Validation dataset file not found and overwrite is 'False'.")

    with open("log.txt", "a") as g:
        train_ds = SavedGalsimBlends(train_ds_file, split, keep_padding=True)
        val_ds = SavedGalsimBlends(val_ds_file, n_samples - split, keep_padding=True)
        train_dl, val_dl, trainer = setup_training_objects(
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            n_epochs=n_epochs,
            validate_every_n_epoch=validate_every_n_epoch,
            val_check_interval=None,
            model_name="deblender",
            log_every_n_steps=50,
            log_file=g,
        )

    # setup model to train
    galaxy_encoder = GalaxyEncoder(AE_STATE_DICT)

    trainer.fit(model=galaxy_encoder, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
