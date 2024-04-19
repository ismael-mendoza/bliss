import numpy as np
import torch

from bliss.render_tiles import (
    fit_source_to_ptile,
    reconstruct_image_from_ptiles,
    size_galaxy,
    trim_source,
)


def test_reconstruct_image_from_ptiles():
    ptiles = torch.randn((32, 10, 10, 1, 52, 52)) * 10 + 100

    images = reconstruct_image_from_ptiles(ptiles, tile_slen=4, bp=24)

    assert images.shape == (32, 1, 88, 88)


def test_trim_source():
    source = torch.randn((1, 53, 53))

    new_source1 = old_trim_source(source, 52)
    new_source = trim_source(source, 52)
    new_source2 = fit_source_to_ptile(source, 52)
    new_source3 = size_galaxy(source[None, :, :, :], 52)

    assert new_source.shape == new_source1.shape
    assert new_source.shape == new_source2.shape
    assert new_source.shape == new_source3[0].shape

    x = 53
    y = 52
    assert x + ((x % 2) == 0) * 1 == 53
    assert y + ((y % 2) == 0) * 1 == 53


def old_trim_source(source, ptile_slen: int):
    """Crop the source to length ptile_slen x ptile_slen, centered at the middle."""
    assert len(source.shape) == 3

    # if self.ptile_slen is even, we still make source dimension odd.
    # otherwise, the source won't have a peak in the center pixel.
    local_slen = ptile_slen + ((ptile_slen % 2) == 0) * 1

    source_slen = source.shape[2]
    source_center = (source_slen - 1) / 2

    assert source_slen >= local_slen

    r = np.floor(local_slen / 2)
    l_indx = int(source_center - r)
    u_indx = int(source_center + r + 1)

    return source[:, l_indx:u_indx, l_indx:u_indx]
