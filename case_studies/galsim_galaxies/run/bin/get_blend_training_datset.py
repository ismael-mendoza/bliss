#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L

from case_studies.galsim_galaxies.run.training_functions import create_dataset

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-n", "--n-samples", default=1280 * 30, type=int)
@click.option("--split", default=1280 * 20, type=int)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
@click.option("--only-bright", is_flag=True, default=False)
@click.option("--no-padding-galaxies", is_flag=True, default=False)
@click.option("--galaxy-density", default=185, type=float)
@click.option("--star-density", default=10, type=float)
def main(
    seed: int,
    n_samples: int,
    split: int,
    tag: str,
    only_bright: bool,
    no_padding_galaxies: bool,
    galaxy_density: float,
    star_density: float,
):

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/train_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/val_ds_{tag}.pt"

    if Path(train_ds_file).exists():
        raise IOError("Training file already exists")

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training blend data generation script...
        With tag {tag} and seed {seed} at {now}
        Galaxy density {galaxy_density}, star_density {star_density}, and
        Only bright '{only_bright}', no padding galaxies '{no_padding_galaxies}'.
        n_samples {n_samples}, split {split}
        """
        print(log_msg, file=f)

    with open("log.txt", "a") as g:
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
            log_file=g,
        )


if __name__ == "__main__":
    main()
