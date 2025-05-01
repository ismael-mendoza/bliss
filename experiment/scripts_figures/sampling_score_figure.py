from functools import partial

import matplotlib.pyplot as plt
import torch
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.io import load_dataset_npz
from bliss.datasets.lsst import APERTURE_BACKGROUND
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import (
    get_blendedness,
    get_deblended_reconstructions,
    get_residual_measurements,
    match_by_grade,
    pred_in_batches,
)


def get_score_dict(
    *,
    images: Tensor,
    truth: FullCatalog,
    tile_cat: TileCatalog,  # could be sampled
    deblender: GalaxyEncoder,
    paddings: Tensor,
    bp: int,
    slen: int,
    aperture: float,
    no_bar: bool = True,
):
    def _pred_fnc(x, y):
        return {"gparams": deblender.variational_mode(x, y)}

    assert "galaxy_bools" in tile_cat.to_dict()

    gparams_dict = pred_in_batches(
        _pred_fnc,
        images,
        tile_cat.locs,
        device=deblender.device,
        batch_size=500,
        no_bar=no_bar,
        desc="Producing galaxy parameters",
    )
    tile_gparams = gparams_dict["gparams"]
    tile_gparams *= tile_cat["galaxy_bools"]
    tile_cat["galaxy_params"] = tile_gparams
    cat = tile_cat.to_full_params()

    # slowest step: deblending
    recon_uncentered = get_deblended_reconstructions(
        cat,
        deblender._dec,
        slen=slen,
        device=deblender.device,
        batch_size=500,
        no_bar=no_bar,
    )

    # measure fluxes
    meas_dict = get_residual_measurements(
        cat,
        images,
        paddings=paddings,
        sources=recon_uncentered,
        bp=bp,
        r=aperture,
        no_bar=no_bar,
    )
    cat["fluxes"] = meas_dict["flux"]
    cat["snr"] = meas_dict["snr"].clip(0)

    # now we can match and compute score
    grades = []
    fluxes1 = []
    fluxes2 = []
    snrs = []
    blds = []
    n_misses = []
    for ii in tqdm(range(len(images)), desc="Computing matches and grades", disable=no_bar):
        locs1 = truth.plocs[ii]
        locs2 = cat.plocs[ii]
        n_sources2 = cat.n_sources[ii].item()
        _fluxes1 = truth["fluxes"][ii][..., 0]
        _fluxes2 = cat["fluxes"][ii][..., 0]
        snr = truth["snr"][ii][..., 0]
        bld = truth["blendedness"][ii][..., 0]

        r, c, dkeep, _ = match_by_grade(
            locs1=locs1, locs2=locs2, fluxes1=_fluxes1, fluxes2=_fluxes2
        )

        if dkeep.sum() > 0:  # at least one matched object
            f1 = _fluxes1[r][dkeep]
            f2 = _fluxes2[c][dkeep]
            _snr = snr[r][dkeep]
            _bld = bld[r][dkeep]
            _grades = (1 + (f1 - f2).abs() / (f1 + APERTURE_BACKGROUND)) ** -1

            # collect data
            grades.append(_grades)

            n_misses.append(n_sources2 - dkeep.sum())
            fluxes1.append(f1)
            fluxes2.append(f2)
            snrs.append(_snr)
            blds.append(_bld)

        else:  # no matched objects
            grades.append(torch.tensor([]))
            n_misses.append(n_sources2)
            fluxes1.append(torch.tensor([]))
            fluxes2.append(torch.tensor([]))
            snrs.append(torch.tensor([]))
            blds.append(torch.tensor([]))

    # pad with zero to ensure all lists are the same length
    def pad_tensor_list(tensor_list, *, max_len: int):
        for t in tensor_list:
            if len(t) > max_len:
                raise ValueError("Tensor list contains tensors longer than max_len")
        padded_list = [
            torch.cat([t, torch.zeros(max_len - len(t))]) if len(t) < max_len else t
            for t in tensor_list
        ]
        return torch.stack(padded_list)

    max_len = max(len(g) for g in grades) if grades else 0
    grades = pad_tensor_list(grades, max_len=max_len)
    fluxes1 = pad_tensor_list(fluxes1, max_len=max_len)
    fluxes2 = pad_tensor_list(fluxes2, max_len=max_len)
    snrs = pad_tensor_list(snrs, max_len=max_len)
    blds = pad_tensor_list(blds, max_len=max_len)

    return {
        "n_misses": torch.tensor(n_misses),
        "n_detections": cat.n_sources,
        "pred_snrs": cat["snr"],
        "grades": grades,
        "fluxes1": fluxes1,
        "fluxes2": fluxes2,
        "true_snrs": snrs,
        "blds": blds,
    }


def get_sep_score_dict():
    """Use SEP for both detection and deblending (with segmentation)."""
    pass


class ScoreFigure(BlissFigure):
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
            "scores": {},
        }

    @property
    def cache_name(self) -> str:
        return "score"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("scores",)

    def compute_data(self, ds_path: str, detection: DetectionEncoder, deblender: GalaxyEncoder):
        device = detection.device

        # metadata
        bp = detection.bp
        tile_slen = detection.tile_slen

        # read dataset
        dataset = load_dataset_npz(ds_path)
        # dataset = {k: v[:100] for k, v in dataset.items()}  # limit to first 100 blends

        images = dataset["images"]
        paddings = dataset["paddings"]
        noiseless = dataset["noiseless"]

        # only galaxies allowed in dataset (for now)
        galaxy_uncentered = dataset["uncentered_sources"]
        n_sources = dataset["n_sources"]
        galaxy_bools = dataset["galaxy_bools"]
        assert torch.all(n_sources.flatten() == galaxy_bools.sum(axis=1).flatten().long())

        # more metadata
        slen = images.shape[-1] - 2 * bp
        nth = (images.shape[2] - 2 * bp) // tile_slen
        ntw = (images.shape[3] - 2 * bp) // tile_slen
        assert slen % tile_slen == 0, "slen must be divisible by tile_slen"

        # get truth catalog
        exclude = ("images", "uncentered_sources", "centered_sources", "noiseless", "paddings")
        true_cat_dict = {p: q for p, q in dataset.items() if p not in exclude}
        truth = FullCatalog(slen, slen, true_cat_dict)

        # add true snr to truth catalog using aperture photometry
        meas_truth = get_residual_measurements(
            truth, images, paddings=paddings, sources=galaxy_uncentered, bp=bp, r=self.aperture
        )

        # we don't need to align here because we are matching later with the predictions
        truth["snr"] = meas_truth["snr"].clip(0)
        truth["blendedness"] = get_blendedness(galaxy_uncentered, noiseless).unsqueeze(-1)
        truth["fluxes"] = meas_truth["flux"]
        truth["ellips"] = meas_truth["ellips"]
        truth["sigma"] = meas_truth["sigma"]

        # first get MAP variational parameters
        def _pred_fnc1(x):
            return detection.variational_mode(x).to_dict()

        map_tile_cat_dict = pred_in_batches(
            _pred_fnc1,
            images,
            device=device,
            desc="Producing MAP catalog",
            no_bar=False,
        )
        map_tile_cat = TileCatalog(tile_slen, map_tile_cat_dict).to("cpu")
        # dataset only contains galaxies
        map_tile_cat["galaxy_bools"] = rearrange(map_tile_cat.n_sources, "b nth ntw -> b nth ntw 1")

        map_results = get_score_dict(
            images=images,
            truth=truth,
            tile_cat=map_tile_cat,
            deblender=deblender,
            paddings=paddings,
            bp=bp,
            slen=slen,
            aperture=self.aperture,
            no_bar=False,
        )

        # now get samples
        _pred_fnc2 = partial(detection.sample, n_samples=self.n_samples)
        samples = pred_in_batches(
            _pred_fnc2,
            images,
            device=device,
            desc="Sampling images (detection)",
            no_bar=False,
            axis=1,
        )

        # set n_sources to 0 if locs are out of bounds, also set locs to 0
        # otherwise `TileCatalog` below breaks
        mask = samples["locs"][..., 0].lt(0) | samples["locs"][..., 0].gt(1)
        mask |= samples["locs"][..., 1].lt(0) | samples["locs"][..., 1].gt(1)
        samples["n_sources"][mask] = 0
        samples["locs"][mask, :] = 0.0

        # get full catalogs from tile catalogs
        tile_cats = []
        for ii in range(self.n_samples):
            tile_cat = TileCatalog.from_flat_dict(
                tile_slen, nth, ntw, {k: v[ii] for k, v in samples.items()}
            )
            tile_cat["galaxy_bools"] = rearrange(tile_cat.n_sources, "b nth ntw -> b nth ntw 1")
            tile_cats.append(tile_cat)

        # compute scores for each sample
        sample_results = []
        for tile_cat in tqdm(tile_cats, desc="Computing scores for samples"):
            sample_results.append(
                get_score_dict(
                    images=images,
                    truth=truth,
                    tile_cat=tile_cat,
                    deblender=deblender,
                    paddings=paddings,
                    bp=bp,
                    slen=slen,
                    aperture=self.aperture,
                    no_bar=True,
                )
            )

        # pad with zero across all samples
        for key in map_results:
            if key not in ("n_misses",):
                max_len = max(res[key].shape[1] for res in sample_results)
                for res in sample_results:
                    val = res[key]
                    if val.shape[1] < max_len:
                        _val = torch.cat(
                            [val, torch.zeros(val.shape[0], max_len - val.shape[1])], dim=1
                        )
                        res[key] = _val
                sample_results[key] = torch.stack([res[key] for res in sample_results])
            else:
                # n_misses is a 1D tensor, so we can just stack them
                sample_results[key] = torch.stack([res[key] for res in sample_results])

        return {
            "map": map_results,
            "samples": sample_results,
            "truth": {
                "snr": truth["snr"],
                "bld": truth["blendedness"],
                "n_sources": truth.n_sources,
            },
        }

    def create_figure(self, fname, data):
        if fname == "scores":
            return plt.figure(figsize=(6, 6))
        raise ValueError(f"Unknown figure name: {fname}")
