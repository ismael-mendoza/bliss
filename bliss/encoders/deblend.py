from pathlib import Path

import pytorch_lightning as pl
import torch
from einops import pack, rearrange, unpack
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.datasets.galsim_blends import parse_dataset
from bliss.encoders.autoencoder import CenteredGalaxyDecoder, CenteredGalaxyEncoder
from bliss.grid import center_ptiles
from bliss.render_tiles import (
    get_images_in_tiles,
    reconstruct_image_from_ptiles,
    render_galaxy_ptiles,
    validate_border_padding,
)


class GalaxyEncoder(pl.LightningModule):
    def __init__(
        self,
        tile_slen: int,
        ptile_slen: int,
        decoder_state_dict: str,
        decoder_slen: int = 53,
        n_bands: int = 1,
        latent_dim: int = 8,
        hidden: int = 256,
        crop_loss_at_border: bool = False,
    ):
        super().__init__()

        # dimensions
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)
        self.final_slen = self.ptile_slen - 2 * self.tile_slen  # will always crop 2 * tile_slen

        self._crop_loss_at_border = crop_loss_at_border

        # encoder (to be trained)
        self._latent_dim = latent_dim
        self._hidden = hidden
        self._enc = CenteredGalaxyEncoder(self.final_slen, latent_dim, n_bands, hidden)

        # decoder
        self._dec = CenteredGalaxyDecoder(decoder_slen, latent_dim, n_bands, hidden)
        self._dec.load_state_dict(
            torch.load(Path(decoder_state_dict), map_location=torch.device("cpu"))
        )
        self._dec.requires_grad_(False)
        self._dec.eval()

    def forward(self, image_ptiles: Tensor, tile_locs: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode(image_ptiles, tile_locs)

    def encode(self, image_ptiles_flat: Tensor, tile_locs_flat: Tensor) -> tuple[Tensor, Tensor]:
        """Runs galaxy encoder on input image ptiles (with bg substracted)."""
        centered_ptiles = self._get_centered_padded_tiles(image_ptiles_flat, tile_locs_flat)
        assert centered_ptiles.shape[-1] == centered_ptiles.shape[-2] == self.final_slen
        return self._enc(centered_ptiles)

    def get_loss(self, images: Tensor, background: Tensor, tile_catalog: TileCatalog):
        _, nth, ntw, _ = tile_catalog.locs

        images_with_background, _ = pack([images, background], "b * h w")
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles_flat = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")
        tile_locs_flat = rearrange(tile_catalog.locs, "n nth ntw xy -> (n nth ntw) xy")
        out: tuple[Tensor, Tensor] = self(image_ptiles_flat, tile_locs_flat)
        galaxy_params_flat, pq_divergence_flat = out
        assert galaxy_params_flat.ndim == 2 and pq_divergence_flat.ndim == 1

        # draw fully reconstructed image.
        # NOTE: Assume recon_mean = recon_var per poisson approximation.
        galaxy_params_pred = rearrange(
            galaxy_params_flat, "(b nth ntw) ms d -> b nth ntw ms d", nth=nth, ntw=ntw
        )
        recon_ptiles = render_galaxy_ptiles(
            self._dec,
            tile_catalog.locs,
            galaxy_params_pred,
            tile_catalog["galaxy_bools"],
            self.ptile_slen,
            self.tile_slen,
            self.n_bands,
        )
        recon_mean = reconstruct_image_from_ptiles(recon_ptiles, self.tile_slen, self.bp)
        recon_mean += background
        assert recon_mean.ndim == 4 and recon_mean.shape[-1] == images.shape[-1]

        assert not torch.any(torch.isnan(recon_mean))
        assert not torch.any(torch.isinf(recon_mean))
        recon_losses = -Normal(recon_mean, recon_mean.sqrt()).log_prob(images)
        if self._crop_loss_at_border:
            bp = self.bp * 2
            recon_losses = recon_losses[:, :, :, bp:(-bp), bp:(-bp)]
        assert not torch.any(torch.isnan(recon_losses))
        assert not torch.any(torch.isinf(recon_losses))

        # For divergence loss, we only evaluate tiles with a galaxy in them
        galaxy_bools = tile_catalog["galaxy_bools"]
        galaxy_bools_flat = rearrange(galaxy_bools, "n nth ntw 1 -> (n nth ntw)")
        divergence_loss = (pq_divergence_flat * galaxy_bools_flat).sum()
        return recon_losses.sum() - divergence_loss

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        images, background, tile_catalog = parse_dataset(batch, self.tile_slen)
        loss = self.get_loss(images, background, tile_catalog)
        self.log("train/loss", loss, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        images, background, tile_catalog = parse_dataset(batch, self.tile_slen)
        loss = self.get_loss(images, background, tile_catalog)
        self.log("val/loss", loss, batch_size=len(images))
        return loss

    def configure_optimizers(self):
        """Set up optimizers."""
        return Adam(self._enc.parameters(), 1e-3)

    def _get_centered_padded_tiles(self, image_ptiles: Tensor, tile_locs_flat: Tensor) -> Tensor:
        img, bg = unpack(image_ptiles, [(1,), (1,)], "b nht nhw * h w")
        return center_ptiles(img - bg, tile_locs_flat, self.tile_slen, self.bp)
