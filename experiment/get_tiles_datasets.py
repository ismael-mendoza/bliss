#!/usr/bin/env python3
"""We generate 3 different datasets for each type of encoder."""

import datetime

import click
import numpy as np
import pytorch_lightning as L

from bliss import DATASETS_DIR, HOME_DIR
from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    STAR_DENSITY,
    get_default_lsst_psf,
    prepare_final_galaxy_catalog,
    prepare_final_star_catalog,
)
from bliss.datasets.padded_tiles import generate_padded_tiles

LOG_FILE = HOME_DIR / "experiment/log.txt"

CATSIM_CAT = prepare_final_galaxy_catalog()
STAR_MAGS = prepare_final_star_catalog()

PSF = get_default_lsst_psf()


assert LOG_FILE.exists()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
@click.option("--n-train", default=50000, type=int)
@click.option("--n-val", default=10000, type=int)
@click.option("--galaxy-density", default=GALAXY_DENSITY, type=float)
@click.option("--star-density", default=STAR_DENSITY, type=float)
def main(seed: int, n_train: int, n_val: int, galaxy_density: float, star_density: float):
    L.seed_everything(seed)

    # galaxies, stars, uncentered, possibly empty tiles
    train_ds_detection_file = DATASETS_DIR / f"train_ds_detection{seed}.npz"
    val_ds_detection_file = DATASETS_DIR / f"val_ds_detection{seed}.npz"

    # galaxies, stars, centered, no empty tiles
    train_ds_binary_file = DATASETS_DIR / f"train_ds_binary{seed}.npz"
    val_ds_binary_file = DATASETS_DIR / f"val_ds_binary{seed}.npz"

    # galxies, centered, no empty tiles
    train_ds_deblend_file = DATASETS_DIR / f"train_ds_deblend{seed}.npz"
    val_ds_deblend_file = DATASETS_DIR / f"val_ds_deblend{seed}.npz"

    assert not train_ds_detection_file.exists(), "files exist"
    assert not val_ds_detection_file.exists(), "files exist"

    # disjointed tables with different galaxies
    indices_fpath = DATASETS_DIR / f"indices_{seed}.npz"
    assert indices_fpath.exists()
    indices_dict = np.load(indices_fpath)
    train_indices = indices_dict["train"]
    val_indices = indices_dict["val"]

    table1 = CATSIM_CAT[train_indices]
    table2 = CATSIM_CAT[val_indices]

    # detection
    ds1 = generate_padded_tiles(
        n_train,
        table1,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
    )
    ds2 = generate_padded_tiles(
        n_val,
        table2,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
    )
    save_dataset_npz(ds1, train_ds_detection_file)
    save_dataset_npz(ds2, val_ds_detection_file)

    # binary
    ds1 = generate_padded_tiles(
        n_train,
        table1,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
        p_source_in=1.0,
    )
    ds2 = generate_padded_tiles(
        n_val,
        table2,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
        p_source_in=1.0,
    )
    save_dataset_npz(ds1, train_ds_binary_file)
    save_dataset_npz(ds2, val_ds_binary_file)

    # deblend
    ds1 = generate_padded_tiles(
        n_train,
        table1,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
        p_source_in=1.0,
        galaxy_prob=1.0,
    )
    ds2 = generate_padded_tiles(
        n_val,
        table2,
        STAR_MAGS,
        psf=PSF,
        max_shift=0.5,
        p_source_in=1.0,
        galaxy_prob=1.0,
    )
    save_dataset_npz(ds1, train_ds_deblend_file)
    save_dataset_npz(ds2, val_ds_deblend_file)

    # logging
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        log_msg = f"""Tile test data generation with seed {seed} at {now}.
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_train}.
        """
        print(log_msg, file=f)


if __name__ == "__main__":
    main()
