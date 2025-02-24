"""Datasets actually used to train models based on cached galaxy iamges."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from bliss.datasets.io import load_dataset_npz


class SavedGalsimBlends(Dataset):
    def __init__(
        self,
        dataset_file: str,
        is_deblender: bool = False,
    ) -> None:
        super().__init__()
        ds: dict[str, Tensor] = load_dataset_npz(dataset_file)

        self.images = ds.pop("images")
        self.epoch_size = len(self.images)

        if not is_deblender:
            ds.pop("uncentered_sources")
            ds.pop("paddings")
            ds.pop("centered_sources")
            self.centered_sources = torch.tensor([0]).float()
        else:
            noise = self.images - ds.pop("uncentered_sources") - ds.pop("paddings")
            centered = ds.pop("centered") + noise
            self.centered = centered
            ds.pop("n_sources")

        ds.pop("galaxy_params")
        ds.pop("star_fluxes")
        ds.pop("star_bools")

        self.tile_params = {**ds}

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "centered": self.centered[index],
            **tile_params_ii,
        }


class SavedIndividualGalaxies(Dataset):
    def __init__(self, dataset_file: str) -> None:
        super().__init__()
        ds: dict[str, Tensor] = load_dataset_npz(dataset_file)

        self.images = ds.pop("images")
        self.epoch_size = len(self.images)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        return {"images": self.images[index]}
