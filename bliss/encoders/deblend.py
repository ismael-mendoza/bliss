from copy import deepcopy

import pytorch_lightning as pl
import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam

from bliss.datasets.lsst import BACKGROUND
from bliss.datasets.padded_tiles import parse_dataset
from bliss.encoders.autoencoder import CenteredGalaxyEncoder, OneCenteredGalaxyAE
from bliss.grid import validate_border_padding
from bliss.render_tiles import get_images_in_tiles


class GalaxyEncoder(pl.LightningModule):
    def __init__(
        self,
        ae_state_dict_path: str,
        n_bands: int = 1,
        tile_slen: int = 5,
        ptile_slen: int = 53,
        latent_dim: int = 8,
        hidden: int = 256,
        lr: float = 1e-4,
    ):
        super().__init__()

        # dimensions
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)

        self.lr = lr

        # encoder (to be trained)
        self._latent_dim = latent_dim
        self._hidden = hidden
        self._enc = CenteredGalaxyEncoder(ptile_slen, latent_dim, n_bands, hidden)

        # decoder
        ae = OneCenteredGalaxyAE(ptile_slen, latent_dim, hidden, n_bands)
        ae.load_state_dict(torch.load(ae_state_dict_path, weights_only=True))
        self._dec = deepcopy(ae.dec)
        self._dec.requires_grad_(False)
        self._dec.eval()
        del ae

        self.register_buffer("background_sqrt", BACKGROUND.sqrt())

    def forward(self, ptiles_flat: Tensor) -> Tensor:
        """Runs galaxy encoder on input image ptiles."""
        return self._enc(ptiles_flat)

    def get_loss(self, ptiles_flat: Tensor, paddings: Tensor):
        galaxy_params_flat: Tensor = self(ptiles_flat)
        recon_mean = self._dec.forward(galaxy_params_flat)
        recon_mean += paddings  # target only galaxies within tiles

        assert recon_mean.ndim == 4 and recon_mean.shape[-1] == ptiles_flat.shape[-1]
        assert not torch.any(torch.logical_or(torch.isnan(recon_mean), torch.isinf(recon_mean)))

        recon_losses: Tensor = -Normal(recon_mean, self.background_sqrt).log_prob(ptiles_flat)
        return recon_losses.sum(), recon_losses.mean(), recon_mean

    def training_step(self, batch, batch_idx):
        """Pytorch lightning training step."""
        ptiles, _, paddings = parse_dataset(batch, self.tile_slen)
        loss, loss_avg, recon = self.get_loss(ptiles, paddings)

        res = (ptiles - recon) / self.background_sqrt
        mean_max_residual = reduce(res.abs(), "b c h w -> b", "max").mean()

        self.log("train/loss", loss, batch_size=len(ptiles), on_step=False, on_epoch=True)
        self.log("train/loss_avg", loss_avg, batch_size=len(ptiles), on_step=False, on_epoch=True)
        self.log(
            "train/mean_max_residual",
            mean_max_residual,
            batch_size=len(ptiles),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning validation step."""
        ptiles, _, paddings = parse_dataset(batch, self.tile_slen)
        loss, loss_avg, recon = self.get_loss(ptiles, paddings)

        res = (ptiles - recon) / self.background_sqrt
        mean_max_residual = reduce(res.abs(), "b c h w -> b", "max").mean()

        self.log("val/loss", loss, batch_size=len(ptiles))
        self.log("val/loss_avg", loss_avg, batch_size=len(ptiles))
        self.log("val/mean_max_residual", mean_max_residual, batch_size=len(ptiles))
        self.log("val/max_residual", res.abs().max(), batch_size=len(ptiles))

        return loss

    def configure_optimizers(self):
        """Set up optimizers."""
        return Adam(self._enc.parameters(), self.lr)

    def variational_mode(self, images: Tensor, tile_locs: Tensor):
        _, nth, ntw, _ = tile_locs.shape

        image_ptiles = get_images_in_tiles(images, self.tile_slen, self.ptile_slen)
        image_ptiles_flat = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")
        tile_locs_flat = rearrange(tile_locs, "n nth ntw xy -> (n nth ntw) xy")
        galaxy_params_flat: Tensor = self(image_ptiles_flat, tile_locs_flat)

        return rearrange(galaxy_params_flat, "(b nth ntw) d -> b nth ntw d", nth=nth, ntw=ntw)
