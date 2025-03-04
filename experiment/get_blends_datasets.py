#!/usr/bin/env python3

import datetime

import click
import numpy as np
import pytorch_lightning as L

from bliss import DATASETS_DIR, HOME_DIR
from bliss.datasets.generate_blends import generate_dataset
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)

LOG_FILE = HOME_DIR / "experiment/log.txt"

CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()


assert LOG_FILE.exists()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("-n", "--n-samples", default=50000, type=int)  # equally divided total blends
@click.option("--galaxy-density", default=GALAXY_DENSITY, type=float)
@click.option("--star-density", default=STAR_DENSITY, type=float)
def main(seed: int, n_samples: int, galaxy_density: float, star_density: float):
    L.seed_everything(seed)

    train_ds_file = DATASETS_DIR / f"train_ds_{seed}.npz"
    val_ds_file = DATASETS_DIR / f"val_ds_{seed}.npz"
    test_ds_file = DATASETS_DIR / f"test_ds_{seed}.npz"

    assert not train_ds_file.exists(), "files exist"
    assert not val_ds_file.exists(), "files exist"
    assert not test_ds_file.exists(), "files exist"

    # disjointed tables with different galaxies
    indices_fpath = DATASETS_DIR / f"indices_{seed}.npz"
    assert indices_fpath.exists()
    indices_dict = np.load(indices_fpath)
    train_indices = indices_dict["train"]
    val_indices = indices_dict["val"]
    test_indices = indices_dict["test"]

    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]
    table3 = CATSIM_CAT[test_indices]

    files = (train_ds_file, val_ds_file, test_ds_file)
    tables = (table1, table2, table3)
    for fpath, t in zip(files, tables, strict=True):
        ds = generate_dataset(
            n_samples,
            t,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=10,
            galaxy_density=galaxy_density,
            star_density=star_density,
            max_shift=0.5,
        )
        save_dataset_npz(ds, fpath)

    # logging
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        log_msg = f"""\nBlend data generation with seed {seed} at {now}.
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_samples}.
        """
        print(log_msg, file=f)


if __name__ == "__main__":
    main()
