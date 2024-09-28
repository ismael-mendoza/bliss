#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L
import torch

from bliss.datasets.generate_individual import generate_individual_dataset
from bliss.datasets.lsst import get_default_lsst_psf, prepare_final_galaxy_catalog

NUM_WORKERS = 0

HOME_DIR = Path(__file__).parent.parent
DATASETS_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/datasets/")
LOG_FILE = HOME_DIR / "experiment/run/log.txt"


CATSIM_CAT = prepare_final_galaxy_catalog()
PSF = get_default_lsst_psf()

TAG = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

assert DATASETS_DIR.exists()
assert LOG_FILE.exists()


@click.command()
@click.option("-s", "--seed", required=True, type=int)
def main(seed: int):

    L.seed_everything(seed)

    train_ds_file = DATASETS_DIR / f"train_ae_ds_{seed}_{TAG}.pt"
    val_ds_file = DATASETS_DIR / f"val_ae_ds_{seed}_{TAG}.pt"
    test_ds_file = DATASETS_DIR / f"test_ae_ds_{seed}_{TAG}.pt"

    with open(LOG_FILE, "a") as f:
        now = datetime.datetime.now()
        print("", file=f)
        log_msg = f"""Run training autoencoder data generation script...
        With seed {seed} at {now}, n_samples {len(CATSIM_CAT)}.
        Train, test, and val divided into 3 parts of same size. A given galaxy only appears once
        across the 3 groups.

        With TAG: {TAG}
        """
        print(log_msg, file=f)

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


if __name__ == "__main__":
    main()
