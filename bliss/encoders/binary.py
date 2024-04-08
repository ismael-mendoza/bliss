import torch
from einops import pack, rearrange
from torch import Tensor, nn
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.datasets.galsim_blends import parse_dataset
from bliss.encoders.layers import ConcatBackgroundTransform, EncoderCNN, make_enc_final
from bliss.grid import center_ptiles
from bliss.render_tiles import get_images_in_tiles, validate_border_padding


class BinaryEncoder(nn.Module):
    def __init__(
        self,
        input_transform: ConcatBackgroundTransform,
        n_bands: int = 1,
        tile_slen: int = 4,
        ptile_slen: int = 52,
        channel: int = 8,
        hidden: int = 128,
        spatial_dropout: float = 0,
        dropout: float = 0,
    ):
        """Encoder which conditioned on other source params returns probability of galaxy vs. star.

        This class implements the binary encoder, which takes in a synthetic image
        along with true locations and source parameters and returns whether each source in that
        image is a star or a galaxy.

        Arguments:
            input_transform: Transformation to apply to input image.
            n_bands: number of bands
            tile_slen: dimension (in pixels) of each tile.
            ptile_slen: dimension (in pixels) of the individual image padded tiles.
            channel: TODO (document this)
            hidden: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_transform = input_transform

        # extract useful info from image_decoder
        self.n_bands = n_bands

        # put image dimensions together
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)
        self.final_slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        dim_enc_conv_out = ((self.final_slen + 1) // 2 + 1) // 2
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(channel * 4 * dim_enc_conv_out**2, hidden, 1, dropout)

    def forward(self, images: Tensor, background: Tensor, locs: Tensor) -> Tensor:
        """Runs the binary encoder on centered_ptiles."""
        flat_locs = rearrange(locs, "n nth ntw xy -> (n nth ntw) xy", xy=2)
        npt, _ = flat_locs.shape

        images_with_background, _ = pack([images, background], "b * h w")
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")

        centered_tiles = self._center_ptiles(image_ptiles, flat_locs)

        # forward to layer shared by all n_sources
        x = rearrange(centered_tiles, "npt c h w -> npt c h w")
        h = self.enc_conv(x)
        h2 = self.enc_final(h)
        galaxy_probs = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(galaxy_probs, "npt -> npt", npt=npt)

    def get_loss(self, images: Tensor, background: Tensor, tile_catalog: TileCatalog):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        n_sources = tile_catalog.n_sources
        locs = tile_catalog.locs
        galaxy_bools = tile_catalog["galaxy_bools"]

        n_sources_flat = rearrange(n_sources, "b ntw ntw -> (b nth ntw)")
        galaxy_bools_flat = rearrange(galaxy_bools, "b nth ntw 1 -> (b nth ntw 1)")

        galaxy_probs_flat = self.forward(images, background, locs)

        # accuracy
        hits = galaxy_probs_flat.ge(0.5).eq(galaxy_bools_flat.bool())
        hits_with_one_source = hits.logical_and(n_sources_flat.eq(1))
        acc = hits_with_one_source.sum() / n_sources_flat.sum()

        # we need to calculate cross entropy loss, only for "on" sources
        raw_loss = BCELoss(reduction="none")(galaxy_probs_flat, galaxy_bools_flat.float())
        return (raw_loss * n_sources_flat.float()).sum(), acc

    def training_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        images, background, tile_catalog = parse_dataset(batch, tile_slen=self.tile_slen)
        loss, acc = self.get_loss(images, background, tile_catalog)
        self.log("train/loss", loss, batch_size=len(images))
        self.log("train/acc", acc, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        images, background, tile_catalog = parse_dataset(batch, tile_slen=self.tile_slen)
        loss, acc = self.get_loss(images, background, tile_catalog)
        self.log("val/loss", loss, batch_size=len(images))
        self.log("val/acc", acc, batch_size=len(images))
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def _center_ptiles(self, image_ptiles: Tensor, tile_locs_flat: Tensor) -> Tensor:
        transformed_ptiles = self.input_transform(image_ptiles)
        assert transformed_ptiles.shape[-1] == transformed_ptiles.shape[-2] == self.ptile_slen
        cropped_ptiles = center_ptiles(transformed_ptiles, tile_locs_flat, self.tile_slen, self.bp)
        assert cropped_ptiles.shape[-1] == cropped_ptiles.shape[-2] == self.final_slen
        return cropped_ptiles
