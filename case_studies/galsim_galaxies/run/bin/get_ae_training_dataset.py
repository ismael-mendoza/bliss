#!/usr/bin/env python3

import datetime
from pathlib import Path

import click
import pytorch_lightning as L
import torch
from astropy.table import Table

from bliss.datasets.galsim_blends import generate_individual_dataset
from bliss.datasets.lsst import get_default_lsst_psf

NUM_WORKERS = 0


@click.command()
@click.option("-s", "--seed", default=42, type=int)
@click.option("-n", "--n-samples", default=1280 * 50, type=int)  # 75% of total catalog
@click.option("-t", "--tag", required=True, type=str, help="Dataset tag")
def main(
    seed: int,
    n_samples: int,
    tag: str,
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
        With tag {tag} and seed {seed} at {now}, n_samples {n_samples}, split {n_samples//2}
        """
        print(log_msg, file=f)

    with open("log.txt", "a") as g:
        catsim_table = Table.read("../../../data/OneDegSq.fits")
        psf = get_default_lsst_psf()
        mask = catsim_table["i_ab"] < 27.3
        new_table = catsim_table[mask]

        dataset = generate_individual_dataset(n_samples, new_table, psf, slen=53)

        # train, val split
        train_ds = {p: q[: n_samples // 2] for p, q in dataset.items()}
        val_ds = {p: q[n_samples // 2 :] for p, q in dataset.items()}

        # now save  data
        torch.save(train_ds, train_ds_file)
        torch.save(val_ds, val_ds_file)


if __name__ == "__main__":
    main()
