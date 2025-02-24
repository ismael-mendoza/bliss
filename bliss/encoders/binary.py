import pytorch_lightning as pl
import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.datasets.padded_tiles import parse_ptiles_dataset
from bliss.encoders.layers import EncoderCNN, make_enc_final
from bliss.render_tiles import validate_border_padding


class BinaryEncoder(pl.LightningModule):
    def __init__(
        self,
        n_bands: int = 1,
        tile_slen: int = 5,
        ptile_slen: int = 53,
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

        # extract useful info from image_decoder
        self.n_bands = n_bands

        # put image dimensions together
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)

        dim_enc_conv_out = ((self.ptile_slenn + 1) // 2 + 1) // 2
        self._enc_conv = EncoderCNN(n_bands, channel, spatial_dropout)
        self._enc_final = make_enc_final(channel * 4 * dim_enc_conv_out**2, hidden, 1, dropout)

    def forward(self, ptiles_flat: Tensor) -> Tensor:
        return self.encode_tiled(ptiles_flat)

    def encode_tiled(self, ptiles_flat: Tensor):
        npt = len(ptiles_flat)
        x = rearrange(ptiles_flat, "npt c h w -> npt c h w")
        h = self._enc_conv(x)
        h2 = self._enc_final(h)
        galaxy_probs = torch.sigmoid(h2).clamp(1e-4, 1 - 1e-4)
        return rearrange(galaxy_probs, "npt 1 -> npt", npt=npt)

    def get_loss(self, ptiles_flat: Tensor, galaxy_bools_flat: Tensor):
        """Return loss, accuracy, binary probabilities, and MAP classifications for given batch."""

        galaxy_probs_flat: Tensor = self(ptiles_flat)

        # accuracy
        # assume every image has a source
        with torch.no_grad():
            hits = galaxy_probs_flat.ge(0.5).eq(galaxy_bools_flat.bool())
            acc = hits.sum() / len(ptiles_flat)

        # we need to calculate cross entropy loss, only for "on" sources
        loss_vec = BCELoss(reduction="none")(galaxy_probs_flat, galaxy_bools_flat.float())

        # as per paper, we sum over tiles and take mean over batches
        loss = reduce(loss_vec, "b -> ", "mean")

        return loss, acc

    def training_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        ptiles, params, _ = parse_ptiles_dataset(batch, tile_slen=self.tile_slen)
        galaxy_bools = params["galaxy_bools"]
        loss, acc = self.get_loss(ptiles, galaxy_bools)
        self.log("train/loss", loss, batch_size=len(ptiles))
        self.log("train/acc", acc, batch_size=len(ptiles))
        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        ptiles, params, _ = parse_ptiles_dataset(batch, tile_slen=self.tile_slen)
        galaxy_bools = params["galaxy_bools"]
        loss, acc = self.get_loss(ptiles, galaxy_bools)
        self.log("val/loss", loss, batch_size=len(ptiles))
        self.log("val/acc", acc, batch_size=len(ptiles))
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)
