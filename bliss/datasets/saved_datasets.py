"""Datasets actually used to train models based on cached galaxy iamges."""

from typing import Optional

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack, rearrange, reduce
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.background import add_noise_and_background, get_constant_background
from bliss.datasets.lsst import (
    GALAXY_DENSITY,
    PIXEL_SCALE,
    STAR_DENSITY,
    convert_mag_to_flux,
    get_default_lsst_background,
)
from bliss.datasets.table_utils import catsim_row_to_galaxy_params


class SavedGalsimBlends(Dataset):
    def __init__(
        self,
        dataset_file: str,
        slen: int = 40,
        tile_slen: int = 4,
    ) -> None:
        super().__init__()
        ds: dict[str, Tensor] = torch.load(dataset_file)

        self.images = ds.pop("images").float()  # needs to be a float for NN
        self.background = ds.pop("background").float()
        self.epoch_size = len(self.images)

        # stars need to be subratected for deblender
        self.stars = ds.pop("star_fields").float()

        # don't need for training
        ds.pop("centered_sources")
        ds.pop("uncentered_sources")
        ds.pop("noiseless")

        full_catalog = FullCatalog(slen, slen, ds)
        tile_catalogs = full_catalog.to_tile_params(tile_slen, ignore_extra_sources=True)
        self.tile_params = tile_catalogs.to_dict()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "background": self.background[index],
            "star_fields": self.stars[index],
            **tile_params_ii,
        }


class SavedIndividualGalaxies(Dataset):
    def __init__(self, dataset_file: str) -> None:
        super().__init__()
        ds: dict[str, Tensor] = torch.load(dataset_file)

        self.images = ds.pop("images").float()  # needs to be a float for NN
        self.background = ds.pop("background").float()

        self.epoch_size = len(self.images)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        return {
            "images": self.images[index],
            "background": self.background[index],
        }
