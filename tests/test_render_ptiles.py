import torch

from bliss.render_tiles import reconstruct_image_from_ptiles


def test_reconstruct_image_from_ptiles():
    ptiles = torch.randn((32, 10, 10, 1, 52, 52)) * 10 + 100

    images = reconstruct_image_from_ptiles(ptiles, tile_slen=4, bp=24)

    assert images.shape == (32, 1, 88, 88)
