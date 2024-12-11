#!/usr/bin/env python3
import warnings
from copy import deepcopy
from pathlib import Path

import click
import pytorch_lightning as pl
import torch

from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.encoders.encoder import Encoder
from experiment.scripts_figures.ae_figures import AutoEncoderFigures
from experiment.scripts_figures.detection_figures import BlendDetectionFigures
from experiment.scripts_figures.toy_figures import ToySeparationFigure

warnings.filterwarnings("ignore", category=FutureWarning)

ALL_FIGS = ("single", "detection", "deblend", "toy")
CACHEDIR = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"


def _load_models(seed: int, ds_seed: int, device):
    """Load models required for producing results."""

    # encoders
    detection = DetectionEncoder().to(device).eval()
    detection.load_state_dict(
        torch.load(f"models/detection_{ds_seed}_{seed}.pt", map_location=device, weights_only=True)
    )
    detection.requires_grad_(False)

    binary = BinaryEncoder().to(device).eval()
    binary.load_state_dict(
        torch.load(f"models/binary_{ds_seed}_{seed}.pt", map_location=device, weights_only=True)
    )
    binary.requires_grad_(False)

    deblender = GalaxyEncoder(f"models/autoencoder_{ds_seed}_{seed}.pt")
    deblender.load_state_dict(
        torch.load(f"models/deblender_{ds_seed}_{seed}.pt", map_location=device, weights_only=True)
    )
    deblender.requires_grad_(False)

    encoder = Encoder(
        detection.eval(),
        binary.eval(),
        deblender.eval(),
        n_images_per_batch=20,
        n_rows_per_batch=30,
    )
    encoder = encoder.to(device)

    # decoder
    ae = OneCenteredGalaxyAE().to(device).eval()
    ae.load_state_dict(torch.load(f"models/autoencoder_{ds_seed}_{seed}.pt", map_location=device))
    decoder = deepcopy(ae.dec)
    decoder.requires_grad_(False)
    decoder = decoder.eval()
    del ae

    return encoder, decoder


def _make_autoencoder_figures(seed: int, ds_seed: int, device, test_file: str, overwrite: bool):
    print("INFO: Creating autoencoder figures...")
    suffix = f"{ds_seed}_{seed}"
    autoencoder = OneCenteredGalaxyAE()
    autoencoder.load_state_dict(torch.load(f"models/autoencoder_{suffix}.pt", weights_only=True))
    autoencoder = autoencoder.to(device).eval()
    autoencoder.requires_grad_(False)

    # arguments for figures
    args = (autoencoder, test_file)

    # create figure classes and plot.
    AutoEncoderFigures(
        n_examples=5,
        overwrite=overwrite,
        figdir="figures",
        cachedir=CACHEDIR,
        suffix=suffix,
    )(*args)


def _make_detection_figure(
    encoder: Encoder, test_file: str, *, aperture: float, suffix: str, overwrite: bool
):
    print("INFO: Creating figures for detection encoder performance simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": "figures",
        "suffix": suffix,
        "cachedir": CACHEDIR,
        "aperture": aperture,
    }
    BlendDetectionFigures(**_init_kwargs)(detection=encoder.detection_encoder, ds_path=test_file)


@click.command()
@click.option("-m", "--mode", required=True, type=click.Choice(ALL_FIGS, case_sensitive=False))
@click.option("-s", "--seed", required=True, type=int, help="Consistent seed used to train models.")
@click.option("--ds-seed", required=True, type=int, help="Seed of training/testing set.")
@click.option("--aperture", default=5.0, type=float, help="Aperture radius for results.")
@click.option("--test-file-single", default="", type=str, help="Dataset file for testing AE.")
@click.option("--test-file-blends", default="", type=str, help="Dataset file for testing Encoders.")
@click.option("-o", "--overwrite", is_flag=True, default=False, help="Whether to overwrite cache.")
def main(
    mode: str,
    seed: int,
    ds_seed: int,
    aperture: float,
    test_file_single: str,
    test_file_blends: str,
    overwrite: bool,
):
    assert mode in ALL_FIGS
    suffix = f"{ds_seed}_{seed}"

    device = torch.device("cuda:0")
    pl.seed_everything(seed)

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if mode == "single":
        assert test_file_single != "" and Path(test_file_single).exists()
        _make_autoencoder_figures(seed, ds_seed, device, test_file_single, overwrite)

    if mode in {"toy", "detection", "deblend", "classification"}:
        encoder, decoder = _load_models(seed, ds_seed, device)

    if mode == "detection":
        assert test_file_blends != "" and Path(test_file_blends).exists()
        _make_detection_figure(
            encoder, test_file_blends, suffix=suffix, overwrite=overwrite, aperture=aperture
        )

    if mode == "toy":
        print("INFO: Creating figures for testing BLISS on pair galaxy toy example.")
        cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
        ToySeparationFigure(
            overwrite=overwrite, figdir="figures", cachedir=cachedir, suffix=suffix
        )(encoder, decoder)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
