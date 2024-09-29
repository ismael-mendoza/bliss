#!/usr/bin/env python3

import datetime

import click
import pytorch_lightning as L
import torch

from bliss import DATASETS_DIR, HOME_DIR
from bliss.datasets.generate_individual import generate_individual_dataset
from bliss.datasets.lsst import get_default_lsst_psf, prepare_final_galaxy_catalog

NUM_WORKERS = 0

LOG_FILE = HOME_DIR / "experiment/log.txt"

CATSIM_CAT = prepare_final_galaxy_catalog()
PSF = get_default_lsst_psf()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
def main(seed: int):

    L.seed_everything(seed)

    train_ds_file = DATASETS_DIR / f"train_ae_ds_{seed}.pt"
    val_ds_file = DATASETS_DIR / f"val_ae_ds_{seed}.pt"
    test_ds_file = DATASETS_DIR / f"test_ae_ds_{seed}.pt"

    n_rows = len(CATSIM_CAT)

    # shuffled because of indices in random.choice
    dataset = generate_individual_dataset(n_rows, CATSIM_CAT, PSF, slen=53, replace=False)

    # train, val, test split
    # no galaxies shared
    train_ds = {p: q[: n_rows // 3] for p, q in dataset.items()}
    val_ds = {p: q[n_rows // 3 : 2 * n_rows // 3] for p, q in dataset.items()}
    test_ds = {p: q[2 * n_rows // 3 :] for p, q in dataset.items()}

    # now save data
    torch.save(train_ds, train_ds_file)
    torch.save(val_ds, val_ds_file)
    torch.save(test_ds, test_ds_file)

    # logging
    with open(LOG_FILE, "a") as f:
        now = datetime.datetime.now()
        log_msg = f"\nRun training autoencoder data generation script with seed {seed} at {now}."
        print(log_msg, file=f)


if __name__ == "__main__":
    main()
