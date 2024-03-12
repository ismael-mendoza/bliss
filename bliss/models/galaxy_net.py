from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn.functional import relu

from bliss.plotting import plot_image


def vae_loss(image: Tensor, recon_mean: Tensor, pq_z: Tensor):
    recon_loss = -Normal(recon_mean, recon_mean.sqrt()).log_prob(image).sum()
    pq_z_loss = -pq_z.sum()
    return recon_loss + pq_z_loss


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

    def forward(self, image, background):
        return self.reconstruct(image, background)

    def reconstruct(self, image: Tensor, background: Tensor) -> Tuple[Tensor, Tensor]:
        """Gets reconstructed image from running through encoder and decoder."""
        z, pq_z = self.enc.forward(image - background)
        recon_mean = self.dec.forward(z)
        return recon_mean + background, pq_z

    def make_encoder(self, slen: int, latent_dim: int, n_bands: int, hidden: int):
        return CenteredGalaxyEncoder(slen, latent_dim, n_bands, hidden)

    def make_decoder(self, slen: int, latent_dim: int, n_bands: int, hidden: int):
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

    def sample(self, image: Tensor, deterministic=True):
        assert deterministic, "CenteredGalaxyEncoder is deterministic"
        return self.features(image)

    def forward(self, image):
        """Encodes galaxy from image."""
        z = self.sample(image)
        return z, torch.tensor(0, device=image.device)


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

    def forward(self, z):
        """Decodes image from latent representation."""
        z = self.fc(z)
        z = z.view(-1, 8, self.min_slen, self.min_slen)
        z = self.deconv(z)
        z = z[:, :, : self.slen, : self.slen]
        assert z.shape[-1] == self.slen and z.shape[-2] == self.slen
        return relu(z)

    def _validate_slen(self):
        slen2 = _conv2d_inv_dim(_conv2d_inv_dim(self.min_slen))
        if slen2 != self.slen:
            raise ValueError(f"The input slen '{self.slen}' is invalid.")


def _conv2d_out_dim(x: int) -> int:
    """Function to figure out dimension of our Conv2D."""
    return (x - 5) // 3 + 1


def _conv2d_inv_dim(x: int) -> int:
    return (x - 1) * 3 + 5


def plot_reconstruction(
    images: np.ndarray,
    recon_mean: np.ndarray,
    residuals: np.ndarray,
    n_examples: int = 10,
    mode: str = "random",
    width: float = 10,
    pad: float = 6.0,
):
    # only plot i band if available, otherwise the highest band given.
    assert images.shape[0] >= n_examples
    assert images.shape[1] == recon_mean.shape[1] == residuals.shape[1] == 1, "1 band only."
    figsize = (width, width * n_examples / 3)
    fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=figsize)

    if mode == "random":
        indices = torch.randint(0, len(images), size=(n_examples,))
    elif mode == "worst":
        # get indices where absolute residual is the largest.
        absolute_residual = residuals.abs().sum(axis=(1, 2, 3))
        indices = absolute_residual.argsort()[-n_examples:]
    else:
        raise NotImplementedError(f"Specified mode '{mode}' has not been implemented.")

    # pick standard ranges for residuals
    vmin_res = residuals[indices].min().item()
    vmax_res = residuals[indices].max().item()

    for i in range(n_examples):
        idx = indices[i]

        ax_true = axes[i, 0]
        ax_recon = axes[i, 1]
        ax_res = axes[i, 2]

        # only add titles to the first axes.
        if i == 0:
            ax_true.set_title("Images $x$", pad=pad)
            ax_recon.set_title(r"Reconstruction $\tilde{x}$", pad=pad)
            ax_res.set_title(r"Residual $\left(x - \tilde{x}\right) / \sqrt{\tilde{x}}$", pad=pad)

        # standarize ranges of true and reconstruction
        image = images[idx, 0].detach().cpu().numpy()
        recon = recon_mean[idx, 0].detach().cpu().numpy()
        residual = residuals[idx, 0].detach().cpu().numpy()
        vmin = min(image.min().item(), recon.min().item())
        vmax = max(image.max().item(), recon.max().item())

        # plot images
        plot_image(fig, ax_true, image, vrange=(vmin, vmax))
        plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
        plot_image(fig, ax_res, residual, vrange=(vmin_res, vmax_res))

    plt.tight_layout()

    return fig
