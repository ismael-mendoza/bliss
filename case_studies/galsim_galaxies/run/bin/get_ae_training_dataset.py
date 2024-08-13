#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L

from case_studies.galsim_galaxies.run.training_functions import create_dataset

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-n", "--n-samples", default=1280 * 100, type=int)  # 75% of total catalog
@click.option("--split", default=1280 * 75, type=int)
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
@click.option("--only-bright", is_flag=True, default=False)
def main(
    seed: int,
    n_samples: int,
    split: int,
    tag: str,
    only_bright: bool,
):

    L.seed_everything(seed)

    train_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/train_ae_ds_{tag}.pt"
    val_ds_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/val_ae_ds_{tag}.pt"

    if Path(train_ds_file).exists():
        raise IOError("Training file already exists")

    with open("log.txt", "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training autoencoder data generation script...
        With tag {tag} and seed {seed} at {now}
        Only bright '{only_bright}', n_samples {n_samples}, split {split}
        """
        print(log_msg, file=f)

    with open("log.txt", "a") as g:
        create_dataset(
            catsim_file="../../../data/OneDegSq.fits",
            stars_mag_file="../../../data/stars_med_june2018.fits",  # UNUSED
            n_samples=n_samples,
            train_val_split=split,
            train_ds_file=train_ds_file,
            val_ds_file=val_ds_file,
            only_bright=only_bright,
            add_galaxies_in_padding=False,
            galaxy_density=1000,  # hack to always have at least 1 galaxy
            star_density=0,
            max_n_sources=1,
            slen=53,
            bp=0,
            max_shift=0,  # centered
            log_file=g,
        )


if __name__ == "__main__":
    main()
