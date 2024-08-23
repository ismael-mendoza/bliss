#!/usr/bin/env python3

from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_dataset, generate_individual_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor
from bliss.reporting import get_snr

HOME_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(__file__).parent / "data" / "data"
cat = Table.read(HOME_DIR / "data/OneDegSq.fits")
CATSIM_TABLE = cat[cat["i_ab"] < 27.3]
star_mags = column_to_tensor(Table.read(HOME_DIR / "data/stars_med_june2018.fits"), "i_ab")
STAR_MAGS = star_mags[star_mags > 20]
PSF = get_default_lsst_psf()


@click.command()
@click.option("--n-samples", default=20000, type=int)
@click.option("-s", "--seed", default=42, type=int)
@click.option("-o", "--overwrite", is_flag=True, default=False)
def main(n_samples: int, seed: int, overwrite: bool):

    pl.seed_everything(seed)

    dataset_file = DATA_DIR / "blends_test.pt"
    if not overwrite and Path(dataset_file).exists():
        raise IOError("File already exists and overwrite flag is 'False'.")

    # https://www.wolframalpha.com/input?i=poisson+distribution+with+mean+3.5
    dataset = generate_dataset(
        n_samples,
        CATSIM_TABLE,
        STAR_MAGS,
        psf=PSF,
        max_n_sources=10,
    )

    torch.save(dataset, dataset_file)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
