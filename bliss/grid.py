"""Functions for working with `torch.nn.functional.grid_sample`."""

import torch
from einops import pack, rearrange, unpack
from torch import Tensor
from torch.nn.functional import grid_sample


def get_mgrid(slen: int, device: torch.device):
    assert slen >= 3
    offset = (slen - 1) / 2
    offsets = torch.arange(-offset, offset + 1, 1, device=device)
    x, y = torch.meshgrid(offsets, offsets, indexing="ij")  # same as numpy default indexing.
    mgrid_not_normalized, _ = pack([y, x], "h w *")
    # normalize to (-1, 1) and scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return (mgrid_not_normalized / offset).float() * (slen - 1) / slen


def swap_locs_columns(locs: Tensor) -> Tensor:
    """Swap the columns of locs to invert 'x' and 'y' with einops!"""
    assert locs.ndim == 2 and locs.shape[1] == 2
    x, y = unpack(locs, [[1], [1]], "b *")
    return pack([y, x], "b *")[0]


def shift_sources_in_ptiles(
    image_ptiles_flat: Tensor, tile_locs_flat: Tensor, tile_slen: int, bp: int, center=False
) -> Tensor:
    """Shift sources at given padded tiles to given locations.

    The keyword `center` controls whether the sources are already centered at the
    padded tile and should be shifted by `tile_locs_flat` (center=False),
    or if the sources are already shifted by that amount and should be 'centered' (center=True).
    Default is `False`.

    """
    npt, _, _, ptile_slen = image_ptiles_flat.shape
    assert ptile_slen == image_ptiles_flat.shape[-2]
    assert bp == (ptile_slen - tile_slen) // 2
    assert tile_locs_flat.shape[0] == npt

    # get new locs to do the shift
    grid = get_mgrid(ptile_slen, image_ptiles_flat.device)
    ptile_locs = (tile_locs_flat * tile_slen + bp) / ptile_slen
    sgn = 1 if center else -1
    offsets_hw = sgn * (torch.tensor(1.0) - 2 * ptile_locs)
    offsets_xy = swap_locs_columns(offsets_hw)
    grid_inflated = rearrange(grid, "h w xy -> 1 h w xy", xy=2, h=ptile_slen)
    offsets_xy_inflated = rearrange(offsets_xy, "npt xy -> npt 1 1 xy", xy=2)
    grid_loc = grid_inflated - offsets_xy_inflated

    return grid_sample(image_ptiles_flat, grid_loc, align_corners=True)
