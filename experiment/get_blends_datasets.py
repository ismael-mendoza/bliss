#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import numpy as np
import pytorch_lightning as L
import torch

from bliss.datasets.generate_blends import generate_dataset
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)

HOME_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/datasets/")
LOG_FILE = HOME_DIR / "experiment/run/log.txt"


CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()

TAG = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

assert DATASETS_DIR.exists()
assert LOG_FILE.exists()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("-n", "--n-samples", default=50_000, type=int)  # equally divided total blends
@click.option("--galaxy-density", default=GALAXY_DENSITY, type=float)
@click.option("--star-density", default=STAR_DENSITY, type=float)
def main(seed: int, n_samples: int, galaxy_density: float, star_density: float):

    L.seed_everything(seed)
    rng = np.random.default_rng(seed)  # for catalog indices

    train_ds_file = DATASETS_DIR / f"train_ds_{seed}_{TAG}.pt"
    val_ds_file = DATASETS_DIR / f"val_ds_{seed}_{TAG}.pt"
    test_ds_file = DATASETS_DIR / f"test_ds_{seed}_{TAG}.pt"

    with open(LOG_FILE, "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training blend data generation script...
        With seed {seed} at {now}
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_samples}.
        Samples will be divided into 3 datasets of {n_samples} # of blends.

        With TAG: {TAG}
        """
        print(log_msg, file=f)

    # disjointed tables with different galaxies
    n_rows = len(CATSIM_CAT)
    shuffled_indices = rng.choice(np.arange(n_rows), size=n_rows, replace=False)
    train_indices = shuffled_indices[: n_rows // 3]
    val_indices = shuffled_indices[n_rows // 3 : n_rows // 3 * 2]
    test_indices = shuffled_indices[n_rows // 3 * 2 :]

    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]
    table3 = CATSIM_CAT[test_indices]

    files = (train_ds_file, val_ds_file, test_ds_file)
    tables = (table1, table2, table3)
    for f, t in zip(files, tables):
        ds = generate_dataset(
            n_samples,
            t,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=10,
            galaxy_density=galaxy_density,
            star_density=star_density,
            slen=40,
            bp=24,
            max_shift=0.5,
        )
        torch.save(ds, f)


if __name__ == "__main__":
    main()
