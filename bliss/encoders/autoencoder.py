from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import relu


class OneCenteredGalaxyAE(nn.Module):
    def __init__(
        self,
        slen: int = 53,
        latent_dim: int = 8,
        hidden: int = 256,
        n_bands: int = 1,
        ckpt: str | Path | None = None,
    ):
        super().__init__()

        self.enc = self.make_encoder(slen, latent_dim, n_bands, hidden)
        self.dec = self.make_decoder(slen, latent_dim, n_bands, hidden)
        self.latent_dim = latent_dim

        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, map_location=self.device))

    def forward(self, image, background) -> Tensor:
        """Gets reconstructed image from running through encoder and decoder."""
        z = self.enc(image - background)
        recon_mean = self.dec(z)
        return recon_mean + background

    def get_loss(self, image: Tensor, background: Tensor) -> Tensor:
        recon_mean = self(image, background)
        return -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()

    def make_encoder(
        self, slen: int, latent_dim: int, n_bands: int, hidden: int
    ) -> "CenteredGalaxyEncoder":
        return CenteredGalaxyEncoder(slen, latent_dim, n_bands, hidden)

    def make_decoder(
        self, slen: int, latent_dim: int, n_bands: int, hidden: int
    ) -> "CenteredGalaxyDecoder":
        return CenteredGalaxyDecoder(slen, latent_dim, n_bands, hidden)


class CenteredGalaxyEncoder(nn.Module):
    """Encodes single galaxies with noise but no background."""

    def __init__(
        self,
        slen: int,
        latent_dim: int,
        n_bands: int,
        hidden: int,
    ):
        super().__init__()

        self.slen = slen
        self.latent_dim = latent_dim

        min_slen = _conv2d_out_dim(_conv2d_out_dim(slen))

        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 4, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, 5, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(1, -1),
            nn.Linear(8 * min_slen**2, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Encodes galaxy from image."""
        return self.features(image)


class CenteredGalaxyDecoder(nn.Module):
    """Reconstructs noiseless galaxies from encoding with no background."""

    def __init__(self, slen=53, latent_dim=8, n_bands=1, hidden=256):
        super().__init__()

        self.slen = slen
        self.n_bands = n_bands
        self.min_slen = _conv2d_out_dim(_conv2d_out_dim(slen))
        self._validate_slen()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 8 * self.min_slen**2),
            nn.LeakyReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 5, stride=3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, n_bands, 5, stride=3),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decodes image from latent representation."""
        z1 = self.fc(z)
        z2 = z1.view(-1, 8, self.min_slen, self.min_slen)
        z3 = self.deconv(z2)
        z4 = z3[:, :, : self.slen, : self.slen]
        assert z4.shape[-1] == self.slen and z4.shape[-2] == self.slen
        return relu(z4)

    def _validate_slen(self) -> None:
        slen2 = _conv2d_inv_dim(_conv2d_inv_dim(self.min_slen))
        if slen2 != self.slen:
            raise ValueError(f"The input slen '{self.slen}' is invalid.")


def _conv2d_out_dim(x: int) -> int:
    """Function to figure out dimension of our Conv2D."""
    return (x - 5) // 3 + 1


def _conv2d_inv_dim(x: int) -> int:
    return (x - 1) * 3 + 5
