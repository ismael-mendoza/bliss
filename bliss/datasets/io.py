"""Methods to save datasets using h5py."""

# useful resource: https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73

from pathlib import Path

import h5py
import torch
from torch import Tensor


def save_dataset_h5py(ds: dict[str, Tensor], fpath: str) -> None:
    assert not Path(fpath).exists(), "overwriting existing ds"
    assert Path(fpath).suffix == ".hdf5"
    for k, v in ds.items():
        with h5py.File(fpath, "a") as f:
            f.create_dataset(k, data=v.numpy())


def load_dataset_h5py(fpath: str) -> dict[str, Tensor]:
    assert Path(fpath).exists(), "file path does not exists"
    assert Path(fpath).suffix == ".hdf5"
    ds = {}
    with h5py.File(fpath, "r") as f:
        for k, v in f.items():
            ds[k] = torch.from_numpy(v[...])
    return ds
