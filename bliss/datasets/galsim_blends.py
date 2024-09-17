from typing import Optional

import galsim
import numpy as np
import torch
from astropy.table import Table
from einops import pack, rearrange, reduce
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.background import add_noise_and_background, get_constant_background
from bliss.datasets.lsst import (
    DEFAULT_SLEN,
    GALAXY_DENSITY,
    PIXEL_SCALE,
    STAR_DENSITY,
    convert_mag_to_flux,
    get_default_lsst_background,
)
from bliss.datasets.table_utils import catsim_row_to_galaxy_params


class SavedGalsimBlends(Dataset):
    def __init__(
        self,
        dataset_file: str,
        slen: int = DEFAULT_SLEN,
        tile_slen: int = 4,
    ) -> None:
        super().__init__()
        ds: dict[str, Tensor] = torch.load(dataset_file)

        self.images = ds.pop("images").float()  # needs to be a float for NN
        self.background = ds.pop("background").float()
        self.epoch_size = len(self.images)

        # stars need to be subratected for deblender
        self.stars = ds.pop("star_fields").float()

        # don't need for training
        ds.pop("centered_sources")
        ds.pop("uncentered_sources")
        ds.pop("noiseless")

        full_catalog = FullCatalog(slen, slen, ds)
        tile_catalogs = full_catalog.to_tile_params(tile_slen, ignore_extra_sources=True)
        self.tile_params = tile_catalogs.to_dict()

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        tile_params_ii = {p: q[index] for p, q in self.tile_params.items()}
        return {
            "images": self.images[index],
            "background": self.background[index],
            "star_fields": self.stars[index],
            **tile_params_ii,
        }


class SavedIndividualGalaxies(Dataset):
    def __init__(self, dataset_file: str) -> None:
        super().__init__()
        ds: dict[str, Tensor] = torch.load(dataset_file)

        self.images = ds.pop("images").float()  # needs to be a float for NN
        self.background = ds.pop("background").float()

        self.epoch_size = len(self.images)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index) -> dict[str, Tensor]:
        return {
            "images": self.images[index],
            "background": self.background[index],
        }


def generate_individual_dataset(
    n_samples: int, catsim_table: Table, psf: galsim.GSObject, slen: int = 53, replace: bool = True
):
    """Like the function below but it only generates individual galaxies, so much faster to run."""

    background = get_constant_background(get_default_lsst_background(), (n_samples, 1, slen, slen))
    params, ids = _sample_galaxy_params(catsim_table, n_samples, n_samples, replace=replace)
    assert params.shape == (n_samples, 11)
    gals = torch.zeros((n_samples, 1, slen, slen))
    for ii in tqdm(range(n_samples)):
        gal = _render_one_galaxy(params[ii], psf, slen, offset=None)
        gals[ii] = gal

    # add noise
    noisy = add_noise_and_background(gals, background)

    return {
        "images": noisy,
        "background": background,
        "noiseless": gals,
        "galaxy_params": params,
        "indices": ids,
    }


def generate_dataset(
    n_samples: int,
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    psf: galsim.GSObject,
    max_n_sources: int,
    galaxy_density: float = GALAXY_DENSITY,  # counts / sq. arcmin
    star_density: float = STAR_DENSITY,  # counts / sq. arcmin
    slen: int = DEFAULT_SLEN,
    bp: int = 24,
    max_shift: float = 0.5,  # within tile, 0.5 -> maximum
) -> dict[str, Tensor]:
    assert slen > bp * 2, f"Need to add back padding galaxies if want slen: {slen}"

    images_list = []
    noiseless_images_list = []
    uncentered_sources_list = []
    centered_sources_list = []
    star_fields_list = []
    params_list = []

    size = slen + 2 * bp

    background = get_constant_background(get_default_lsst_background(), (n_samples, 1, size, size))

    # internal region, no galaxies in padding
    mean_sources = (galaxy_density + star_density) * (slen * PIXEL_SCALE / 60) ** 2
    galaxy_prob = galaxy_density / (galaxy_density + star_density)

    for ii in tqdm(range(n_samples)):
        full_cat = sample_full_catalog(
            catsim_table,
            all_star_mags,
            mean_sources=mean_sources,
            max_n_sources=max_n_sources,
            slen=slen,
            max_shift=max_shift,
            galaxy_prob=galaxy_prob,
        )
        noiseless, uncentered_sources, centered_sources = render_full_catalog(
            full_cat, psf, slen, bp
        )

        noisy = add_noise_and_background(noiseless, background[ii, None])

        images_list.append(noisy)
        noiseless_images_list.append(noiseless)
        uncentered_sources_list.append(uncentered_sources)
        centered_sources_list.append(centered_sources)
        params_list.append(full_cat.to_tensor_dict())

        # separately keep stars since it's needed in the deblender loss function
        sbool = rearrange(full_cat["star_bools"], "1 ms 1 -> ms 1 1 1")
        all_stars = reduce(uncentered_sources * sbool, "ms 1 h w -> 1 h w", "sum")
        star_fields_list.append(all_stars)

    images, _ = pack(images_list, "* c h w")
    noiseless, _ = pack(noiseless_images_list, "* c h w")
    centered_sources, _ = pack(centered_sources_list, "* n c h w")
    uncentered_sources, _ = pack(uncentered_sources_list, "* n c h w")
    star_fields, _ = pack(star_fields_list, "* c h w")
    paramss = torch.cat(params_list, dim=0)

    assert centered_sources.shape[:3] == (n_samples, max_n_sources, 1)
    assert uncentered_sources.shape[:3] == (n_samples, max_n_sources, 1)

    return {
        "images": images,
        "background": background,
        "noiseless": noiseless,
        "uncentered_sources": uncentered_sources,
        "centered_sources": centered_sources,
        "star_fields": star_fields,
        **paramss,
    }


def parse_dataset(dataset: dict[str, Tensor], tile_slen: int = 4):
    """Parse dataset into a tuple of (images, background, TileCatalog)."""
    params = dataset.copy()  # make a copy to not change argument.
    images = params.pop("images")
    background = params.pop("background")
    star_fields = params.pop("star_fields")
    return images, background, TileCatalog(tile_slen, params), star_fields


def sample_source_params(
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    slen: int = 100,
    max_shift: float = 0.5,
    galaxy_prob: float = 0.9,
) -> dict[str, Tensor]:
    """Returns source parameters corresponding to a single blend."""
    n_sources = _sample_poisson_n_sources(mean_sources, max_n_sources)
    params, _ = _sample_galaxy_params(catsim_table, n_sources, max_n_sources)
    assert params.shape == (max_n_sources, 11)

    star_fluxes = _sample_star_fluxes(all_star_mags, n_sources, max_n_sources)

    galaxy_bools = torch.zeros(max_n_sources, 1)
    star_bools = torch.zeros(max_n_sources, 1)
    galaxy_bools[:n_sources, :] = _bernoulli(galaxy_prob, n_sources)[:, None]
    star_bools[:n_sources, :] = 1 - galaxy_bools[:n_sources, :]

    locs = torch.zeros(max_n_sources, 2)
    locs[:n_sources, 0] = _uniform(-max_shift, max_shift, n_sources) + 0.5
    locs[:n_sources, 1] = _uniform(-max_shift, max_shift, n_sources) + 0.5
    plocs = locs * slen

    return {
        "n_sources": torch.tensor([n_sources]),
        "plocs": plocs,
        "galaxy_bools": galaxy_bools,
        "star_bools": star_bools,
        "galaxy_params": params * galaxy_bools,
        "star_fluxes": star_fluxes * star_bools,
        "fluxes": params[:, -1, None] * galaxy_bools + star_fluxes * star_bools,
    }


def _sample_star_fluxes(all_star_mags: np.ndarray, n_sources: int, max_n_sources: int):
    star_fluxes = torch.zeros((max_n_sources, 1))
    star_mags = np.random.choice(all_star_mags, size=(n_sources,), replace=True)
    star_fluxes[:n_sources, 0] = convert_mag_to_flux(torch.from_numpy(star_mags))
    return star_fluxes


def sample_full_catalog(
    catsim_table: Table,
    all_star_mags: np.ndarray,  # i-band
    mean_sources: float,
    max_n_sources: int,
    slen: int = 40,
    max_shift: float = 0.5,
    galaxy_prob: float = 0.9,
):
    params = sample_source_params(
        catsim_table, all_star_mags, mean_sources, max_n_sources, slen, max_shift, galaxy_prob
    )

    for p, q in params.items():
        if p != "n_sources":
            params[p] = rearrange(q, "n d -> 1 n d")

    return FullCatalog(slen, slen, params)


def render_full_catalog(full_cat: FullCatalog, psf: galsim.GSObject, slen: int, bp: int):
    size = slen + 2 * bp
    full_plocs = full_cat.plocs
    b, max_n_sources, _ = full_plocs.shape
    assert b == 1, "Only one batch supported for now."

    image = torch.zeros(1, size, size)
    centered_noiseless = torch.zeros(max_n_sources, 1, size, size)
    uncentered_noiseless = torch.zeros(max_n_sources, 1, size, size)

    n_sources = int(full_cat.n_sources.item())
    galaxy_params = full_cat["galaxy_params"][0]
    star_fluxes = full_cat["star_fluxes"][0]
    galaxy_bools = full_cat["galaxy_bools"][0]
    star_bools = full_cat["star_bools"][0]
    plocs = full_plocs[0]
    for ii in range(n_sources):
        offset_x = plocs[ii][1] + bp - size / 2
        offset_y = plocs[ii][0] + bp - size / 2
        offset = torch.tensor([offset_x, offset_y])
        if galaxy_bools[ii] == 1:
            source_uncentered = _render_one_galaxy(galaxy_params[ii], psf, size, offset)
            source_centered = _render_one_galaxy(galaxy_params[ii], psf, size, offset=None)
        elif star_bools[ii] == 1:
            source_uncentered = _render_one_star(psf, star_fluxes[ii][0].item(), size, offset)
            source_centered = _render_one_star(psf, star_fluxes[ii][0].item(), size, offset=None)
        else:
            continue
        centered_noiseless[ii] = source_centered
        uncentered_noiseless[ii] = source_uncentered
        image += source_uncentered

    return image, uncentered_noiseless, centered_noiseless


def _sample_galaxy_params(
    catsim_table: Table, n_galaxies: int, max_n_sources: int, replace: bool = True
) -> tuple[Tensor, Tensor]:
    indices = np.random.choice(np.arange(len(catsim_table)), size=(n_galaxies,), replace=replace)

    rows = catsim_table[indices]
    mags = torch.from_numpy(rows["i_ab"].value.astype(np.float32))  # byte order
    gal_flux = convert_mag_to_flux(mags)
    rows["flux"] = gal_flux.numpy().astype(np.float32)

    ids = torch.from_numpy(rows["galtileid"].value.astype(int))
    return catsim_row_to_galaxy_params(rows, max_n_sources), ids


def _render_one_star(
    psf: galsim.GSObject, flux: float, size: int, offset: Optional[Tensor] = None
) -> Tensor:
    assert offset is None or offset.shape == (2,)
    star = psf.withFlux(flux)
    offset = offset if offset is None else offset.numpy()
    image = star.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset)
    return rearrange(torch.from_numpy(image.array), "h w -> 1 h w")


def _render_one_galaxy(
    galaxy_params: Tensor, psf: galsim.GSObject, size: int, offset: Optional[Tensor] = None
) -> Tensor:
    assert offset is None or offset.shape == (2,)
    assert galaxy_params.device == torch.device("cpu") and galaxy_params.shape == (11,)
    fnb, fnd, fnagn, ab, ad, bb, bd, pab, pad, _, total_flux = galaxy_params.numpy()  # noqa:WPS236

    disk_flux = total_flux * fnd / (fnd + fnb + fnagn)
    bulge_flux = total_flux * fnb / (fnd + fnb + fnagn)

    components = []
    if disk_flux > 0:
        assert bd > 0 and ad > 0 and pad > 0
        disk_q = bd / ad
        disk_hlr_arcsecs = np.sqrt(ad * bd)
        disk = galsim.Exponential(flux=disk_flux, half_light_radius=disk_hlr_arcsecs).shear(
            q=disk_q,
            beta=pad * galsim.degrees,
        )
        components.append(disk)
    if bulge_flux > 0:
        assert bb > 0 and ab > 0 and pab > 0
        bulge_q = bb / ab
        bulge_hlr_arcsecs = np.sqrt(ab * bb)
        bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr_arcsecs).shear(
            q=bulge_q,
            beta=pab * galsim.degrees,
        )
        components.append(bulge)
    galaxy = galsim.Add(components)
    gal_conv = galsim.Convolution(galaxy, psf)
    offset = offset if offset is None else offset.numpy()
    galaxy_image = gal_conv.drawImage(nx=size, ny=size, scale=PIXEL_SCALE, offset=offset).array
    return rearrange(torch.from_numpy(galaxy_image), "h w -> 1 h w")


def _sample_poisson_n_sources(mean_sources: float, max_n_sources: int | float) -> int:
    n_sources = torch.distributions.Poisson(mean_sources).sample([1])
    return int(torch.clamp(n_sources, max=torch.tensor(max_n_sources)))


def _uniform(a, b, n_samples=1) -> Tensor:
    """Uses pytorch to return a single float ~ U(a, b)."""
    return (a - b) * torch.rand(n_samples) + b


def _bernoulli(prob, n_samples=1) -> Tensor:
    prob_list = [float(prob) for _ in range(n_samples)]
    return torch.bernoulli(torch.tensor(prob_list))
