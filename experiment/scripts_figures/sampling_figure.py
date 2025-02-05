import matplotlib.pyplot as plt
import numpy as np
import sep_pjw as sep
import torch
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import (
    get_blendedness,
    get_deblended_reconstructions,
    get_residual_measurements,
    match_by_locs,
)


class SamplingFigure(BlissFigure):
    """Create figures related to sampling centroids and fluxes on the full dataset."""

    def __init__(
        self,
        *,
        figdir,
        cachedir,
        suffix,
        overwrite=False,
        img_format="png",
        aperture=5.0,
        n_samples=100,
    ):
        super().__init__(
            figdir=figdir,
            cachedir=cachedir,
            suffix=suffix,
            overwrite=overwrite,
            img_format=img_format,
        )

        self.aperture = aperture
        self.n_samples = n_samples

    @property
    def all_rcs(self):
        return {
            "samples_residual": {},
        }

    @property
    def cache_name(self) -> str:
        return "sampling"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("samples_residual",)

    def compute_data(self, ds_path: str, detection: DetectionEncoder, deblender: GalaxyEncoder):
        device = detection.device

        # metadata
        bp = detection.bp
        tile_slen = detection.tile_slen

        # read dataset
        dataset = load_dataset_npz(ds_path)
        _images = dataset["images"]
        _paddings = dataset["paddings"]
        noiseless = dataset["noiseless"]

        uncentered_sources = dataset["uncentered_sources"]
        galaxy_bools = dataset["galaxy_bools"]
        _tgbools = rearrange(galaxy_bools, "n ms 1 -> n ms 1 1 1 ")
        galaxy_uncentered = uncentered_sources * _tgbools

        # we want to exclude stars only but keep the padding for starters
        star_bools = dataset["star_bools"]
        only_stars = uncentered_sources * rearrange(star_bools, "b n 1 -> b n 1 1 1").float()
        all_stars = reduce(only_stars, "b n c h w -> b c h w", "sum")
        images = _images - all_stars
        paddings = _paddings - all_stars

        # more metadata
        slen = images.shape[-1] - 2 * bp
        nth = (images.shape[2] - 2 * bp) // tile_slen
        ntw = (images.shape[3] - 2 * bp) // tile_slen

        # get truth catalog
        exclude = ("images", "uncentered_sources", "centered_sources", "noiseless", "paddings")
        true_cat_dict = {p: q for p, q in dataset.items() if p not in exclude}
        truth = FullCatalog(slen, slen, true_cat_dict)

        # add true snr to truth catalog using sep
        meas_truth = get_residual_measurements(
            truth, images, paddings, galaxy_uncentered, bp=bp, r=self.aperture
        )
        truth["snr"] = meas_truth["snr"].clip(0)
        truth["blendedness"] = get_blendedness(galaxy_uncentered, noiseless).unsqueeze(-1)

        # we don't need to align here because we are matching later with the predictions

        # first get variational parameters that will be useful for diagnostics
        n_source_probs, locs_mean, locs_sd = detection.forward(images.to(device))
        additional_info = {
            "n_source_probs": n_source_probs.cpu(),
            "locs_mean": locs_mean.cpu(),
            "locs_sd": locs_sd.cpu(),
        }

        # now we sample location first
        samples = detection.sample(images.to(device), n_samples=self.n_samples)
        samples = {k: v.to("cpu") for k, v in samples.items()}

        # set n_sources to 0 if locs are out of bounds, also set locs to 0
        # otherwise `TileCatalog` below breaks
        mask = samples["locs"][..., 0].lt(0) | samples["locs"][..., 0].gt(1)
        mask |= samples["locs"][..., 1].lt(0) | samples["locs"][..., 1].gt(1)
        samples["n_sources"][mask] = 0
        samples["locs"][mask, :] = 0.0

        # get full catalogs from tile catalogs
        cats = []
        n_samples = 100
        for ii in range(n_samples):
            tile_cat = TileCatalog.from_flat_dict(
                tile_slen, nth, ntw, {k: v[ii] for k, v in samples.items()}
            )
            cat = tile_cat.to_full_params()
            cats.append(cat)

        # we need to do some sort of matching for the galaxy we are targeting
        all_fluxes = []
        all_ellips = []
        all_sigmas = []
        for cat in tqdm(cats, total=len(cats), desc="Creating reconstructions for each sample"):

            # stars are excluded from all images, imopse that all detections are galaxies

            tile_cat = cat.to_tile_params(tile_slen)
            tile_cat["galaxy_bools"] = rearrange(tile_cat.n_sources, "b x y -> b x y 1")
            _tile_locs = tile_cat.locs.to(device)
            _tile_gparams = deblender.variational_mode(images.to(device), _tile_locs).to("cpu")
            _tile_gparams *= tile_cat["galaxy_bools"]
            tile_cat["galaxy_params"] = _tile_gparams
            new_cat = tile_cat.to_full_params()

            recon_uncentered = get_deblended_reconstructions(
                new_cat, deblender._dec, slen=slen, device=deblender.device, batch_size=500
            )
            meas = get_residual_measurements(
                new_cat,
                images,
                paddings=torch.zeros_like(images),
                sources=recon_uncentered,
            )

            all_fluxes.append(meas["fluxes"])
            all_ellips.append(meas["ellips"])
            all_sigmas.append(meas["sigma"])

        # now we match
        fluxes = []
        ellips = []
        sigmas = []

        for ii in tqdm(range(len(cats)), desc="Matching catalogs witht truth"):  # read: samples
            f = torch.zeros((images.shape[0], truth.max_n_sources, 1))
            e = torch.zeros((images.shape[0], truth.max_n_sources, 2))
            s = torch.zeros((images.shape[0], truth.max_n_sources, 1))

            for jj in range(images.shape[0]):

                tns = truth.n_sources[jj].items()
                _tplocs = truth.plocs[jj]
                _eplocs = cats[ii].plocs[jj]

                f[jj, :tns, :] = torch.nan
                e[jj, :tns, :] = torch.nan
                s[jj, :tns, :] = torch.nan

                # example: HSC -> 3 pixels for matching (Yr3 Li et al. ~2021)
                tm, em, dkeep, _ = match_by_locs(_tplocs, _eplocs, slack=3)

                for kk in range(len(tm)):
                    if dkeep[kk].item():
                        f[jj, tm[kk], 0] = all_fluxes[ii][jj][em[kk]].item()
                        e[jj, tm[kk], :] = all_ellips[ii][jj][em[kk]]
                        s[jj, tm[kk], 0] = all_sigmas[ii][jj][em[kk]].item()

            fluxes.append(f)
            ellips.append(e)
            sigmas.append(s)

        fluxes = torch.stack(fluxes)
        ellips = torch.stack(ellips)
        sigmas = torch.stack(sigmas)

        assert fluxes.shape[0] == len(cats)

        fluxes_flat = []
        ellips_flat = []
        sigmas_flat = []

        # now let's flatten across the true objects in each image
        for ii in range(len(cats)):
            f = []
            e = []
            s = []
            for jj in range(images.shape[0]):
                nt = truth.n_sources[jj].item()
                for kk in range(nt):
                    f.append(fluxes[ii, jj, kk, 0].item())
                    e.append(ellips[ii, jj, kk, :])
                    s.append(sigmas[ii, jj, kk, 0].item())
            fluxes_flat.append(f)
            ellips_flat.append(e)
            sigmas_flat.append(s)

        flat_fluxes = torch.tensor(fluxes_flat).unsqueeze(-1)
        ellips_flat = torch.tensor(ellips_flat)
        sigmas_flat = torch.tensor(sigmas_flat).unsqueeze(-1)

        assert flat_fluxes.shape[:1] == (images.shape[0], len(cats))
        assert flat_fluxes.shape[-1] == 1
        assert flat_fluxes.ndim == 4
        assert ellips_flat.shape[-1] == 2

        return {
            "fluxes": flat_fluxes,
            "ellips": ellips_flat,
            "sigmas": sigmas_flat,
            **additional_info,
        }

    def create_figure(self, fname, data):
        if fname == "samples_residual":
            return plt.figure(figsize=(6, 6))
        raise ValueError(f"Unknown figure name: {fname}")
