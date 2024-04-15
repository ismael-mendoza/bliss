#!/usr/bin/env python3

from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor

HOME_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
cat = Table.read(HOME_DIR / "data/OneDegSq.fits")
CATSIM_TABLE = cat[cat["i_ab"] < 27.3]
STAR_MAGS = column_to_tensor(Table.read(HOME_DIR / "data/stars_med_june2018.fits"), "i_ab")
PSF = get_default_lsst_psf()
SEED = 1

pl.seed_everything(SEED)


@click.command()
@click.option("--n-samples", default=10000, type=int)
@click.option("-s", "--seed", default=1, type=int)
@click.option("--mode", type=str, required=True)
def main(n_samples: int, seed: int, mode: str):
    assert mode in {"single", "blend"}

    pl.seed_everything(seed)

    if mode == "single":
        dataset = generate_dataset(
            n_samples,
            CATSIM_TABLE,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=1,
            galaxy_density=1000,
            star_density=0,
            slen=53,
            bp=0,
            max_shift=0,
            add_galaxies_in_padding=False,
        )

        torch.save(dataset, DATA_DIR / "single_galaxies_test.pt")

    else:
        dataset = generate_dataset(
            n_samples,
            CATSIM_TABLE,
            STAR_MAGS,
            psf=PSF,
            max_n_sources=15,
        )

        torch.save(dataset, DATA_DIR / "blends_test.pt")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
