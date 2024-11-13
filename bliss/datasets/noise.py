import torch
from torch import Tensor

from bliss.datasets.lsst import get_default_lsst_background

BACKGROUND = torch.tensor(get_default_lsst_background())


def add_noise(image: Tensor) -> Tensor:
    return image + BACKGROUND.sqrt() * torch.randn_like(image)
