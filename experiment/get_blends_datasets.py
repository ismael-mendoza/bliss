#!/usr/bin/env python3

import datetime

import numpy as np
import pytorch_lightning as L
import typer

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


def main(
    seed: int = typer.Option(),
    n_samples: int = 10000,
    galaxy_density: float = GALAXY_DENSITY,
    star_density: float = STAR_DENSITY,
):
    L.seed_everything(seed)
    test_ds_file = DATASETS_DIR / f"test_ds_{seed}.npz"
    assert not test_ds_file.exists(), "files exist"

    # disjointed tables with different galaxies
    indices_fpath = DATASETS_DIR / f"indices_{seed}.npz"
    assert indices_fpath.exists()
    indices_dict = np.load(indices_fpath)
    test_indices = indices_dict["test"]

    table = CATSIM_CAT[test_indices]

    ds = generate_dataset(
        n_samples,
        table,
        STAR_MAGS,
        psf=PSF,
        max_n_sources=10,
        galaxy_density=galaxy_density,
        star_density=star_density,
        max_shift=0.5,
    )
    save_dataset_npz(ds, test_ds_file)

    # logging
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        log_msg = f"""\nBlend test data generation with seed {seed} at {now}.
        Galaxy density {galaxy_density}, star_density {star_density}, and n_samples {n_samples}.
        """
        print(log_msg, file=f)


if __name__ == "__main__":
    typer.run(main)
