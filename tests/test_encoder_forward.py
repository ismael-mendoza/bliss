"""Test that we can run one loop of encoder training."""

import torch
from astropy.table import Table
from torch.utils.data import DataLoader

from bliss.datasets.io import save_dataset_npz
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.padded_tiles import generate_padded_tiles
from bliss.datasets.saved_datasets import SavedGalsimBlends
from bliss.datasets.table_utils import column_to_tensor
from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder


def test_encoder_forward(home_dir, tmp_path):
    ae_state_dict = home_dir / "experiment" / "models" / "autoencoder_42_42.pt"

    catsim_table = Table.read(home_dir / "data" / "OneDegSq.fits")
    all_star_mags = column_to_tensor(
        Table.read(home_dir / "data" / "stars_med_june2018.fits"), "i_ab"
    )
    psf = get_default_lsst_psf()
    padded_ds = generate_padded_tiles(10, catsim_table, all_star_mags, psf)

    saved_ds_path = tmp_path / "train_ds.npz"
    save_dataset_npz(padded_ds, saved_ds_path)
    saved_ds1 = SavedGalsimBlends(saved_ds_path, is_deblender=False)
    saved_ds2 = SavedGalsimBlends(saved_ds_path, is_deblender=True)

    dl1 = DataLoader(saved_ds1, batch_size=32, num_workers=0)
    dl2 = DataLoader(saved_ds2, batch_size=32, num_workers=0)

    binary_encoder = BinaryEncoder()
    detection_encoder = DetectionEncoder()
    galaxy_encoder = GalaxyEncoder(ae_state_dict)

    with torch.no_grad():
        for b in dl1:
            binary_encoder.get_loss(b["images"], b["galaxy_bools"])
            detection_encoder.get_loss(b["images"], b["n_sources"], b["locs"])

        for b in dl2:
            galaxy_encoder.get_loss(b["images"], b["centered"], b["locs"])
