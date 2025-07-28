#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import typer
from einops import rearrange
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from bliss.catalog import FullCatalog, turn_samples_into_catalogs
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import CLR_CYCLE, binned_statistic, equal_sized_bin_statistic, set_rc_params
from bliss.reporting import (
    get_blendedness,
    get_deblended_reconstructions,
    get_residual_measurements,
    get_sep_catalog,
)
from experiment import CACHE_DIR, DATASETS_DIR, FIGURE_DIR, MODELS_DIR


def _get_sample_results(
    *,
    sorted_indices: np.ndarray,
    n_samples: int,
    images: torch.Tensor,
    paddings: torch.Tensor,
    detection: DetectionEncoder,
    deblender: GalaxyEncoder,
    device: torch.device,
    slen: int,
    tile_slen: int,
    bp: int,
    match_slack: float = 2.0,
) -> list[dict]:
    nth = slen // tile_slen
    assert nth * tile_slen == slen, "Tile size must evenly divide the image size."
    outs = []
    for ii in tqdm(sorted_indices, desc="Processing images"):
        out = {}
        image = images[ii, None]
        padding = paddings[ii, None]

        det_prob, _, _ = detection.forward(image.to(device))
        det_prob = det_prob.cpu()

        samples = detection.sample(image.to(device), n_samples=n_samples)
        samples = {k: v.to("cpu") for k, v in samples.items()}

        # only consider locs in central tile and remove zero ones
        _locs = samples["locs"][:, nth**2 // 2, :].flatten()
        nonzero_locs = _locs[_locs.nonzero()]
        out["nonzero_locs"] = nonzero_locs

        # now get cats
        # this function atuomatically zeroes out sources which locs out of tile
        tile_cats = turn_samples_into_catalogs(samples, tile_slen=tile_slen, nth=nth, ntw=nth)

        # add galaxy params to each catalog
        for jj in tqdm(range(n_samples), desc="Adding galaxy params to catalogs", disable=True):
            _galaxy_bools = rearrange(tile_cats[jj].n_sources, "n nth ntw-> n nth ntw 1")
            tile_cats[jj]["galaxy_bools"] = _galaxy_bools.float()
            _tile_locs = tile_cats[jj].locs
            galaxy_params = deblender.variational_mode(image.to(device), _tile_locs.to(device))
            galaxy_params = galaxy_params.cpu()
            galaxy_params *= tile_cats[jj]["galaxy_bools"]
            tile_cats[jj]["galaxy_params"] = galaxy_params

        # get full cats
        sample_cats = []
        for kk in range(len(tile_cats)):
            sample_cats.append(tile_cats[kk].to_full_params())

        reconstructions = []
        for rr in tqdm(range(n_samples), desc="Reconstructing samples", disable=True):
            recon_uncentered = get_deblended_reconstructions(
                sample_cats[rr],
                deblender._dec,
                slen=slen,
                device=device,
            )
            reconstructions.append(recon_uncentered)

        residual_meas = []
        for ll in tqdm(range(n_samples), desc="Calculating residual measurements", disable=True):
            meas = get_residual_measurements(
                sample_cats[ll],
                image,
                paddings=padding,
                sources=reconstructions[ll],
            )
            assert meas["flux"].shape[0] == 1
            residual_meas.append(meas)

        # pick fluxes that are within central tile only (i.e. that match with central galaxy)
        sample_fluxes = []
        sample_fluxerrs = []
        for ss in range(n_samples):
            meas = residual_meas[ss]
            _plocs = sample_cats[ss].plocs
            assert _plocs.shape[0] == 1 and _plocs.shape[-1] == 2
            plocs = _plocs[0]
            central_plocs = torch.tensor([slen / 2, slen / 2]).reshape(1, 2)
            dist_to_center = torch.norm(plocs - central_plocs, dim=-1)

            # NOTE: match within 2 pixels of center
            indices = torch.argwhere(dist_to_center < match_slack).flatten()
            if len(indices) > 1:
                raise ValueError("More than one source within central tile found.")
            elif len(indices) == 0:
                sample_fluxes.append(torch.nan)
                sample_fluxerrs.append(torch.nan)
            else:
                _idx = indices.item()
                sample_fluxes.append(meas["flux"][0, _idx, 0].item())
                sample_fluxerrs.append(meas["fluxerr"][0, _idx, 0].item())
        sample_fluxes = torch.tensor(sample_fluxes)
        sample_fluxerrs = torch.tensor(sample_fluxerrs)

        n_sources_samples = torch.tensor([cat.n_sources.item() for cat in sample_cats])

        sample_plocs = []
        for ss in range(n_samples):
            _plocs = sample_cats[ss].plocs[0]
            _n_sources = sample_cats[ss].n_sources.item()
            assert _plocs.shape[0] == _n_sources  # only adding nonzero
            sample_plocs.append(_plocs)
        sample_plocs = torch.concatenate(sample_plocs, dim=0)

        out["sample_plocs"] = sample_plocs
        out["n_sources_samples"] = n_sources_samples
        out["det_prob"] = det_prob.reshape(nth, nth).cpu()
        out["sample_fluxes"] = sample_fluxes
        out["sample_fluxerrs"] = sample_fluxerrs
        out["idx"] = ii

        # get map prediction too
        map_tile_cat = detection.variational_mode(image.to(device))
        map_galaxy_bools = rearrange(map_tile_cat.n_sources, "n nth ntw-> n nth ntw 1").float()
        map_tile_cat["galaxy_bools"] = map_galaxy_bools
        map_galaxy_params = deblender.variational_mode(
            image.to(device), map_tile_cat.locs.to(device)
        )
        map_tile_cat["galaxy_params"] = map_galaxy_params * map_galaxy_bools
        map_tile_cat = map_tile_cat.to("cpu")
        map_cat = map_tile_cat.to_full_params()
        map_reconstructions = get_deblended_reconstructions(
            map_cat,
            deblender._dec,
            slen=slen,
            bp=bp,
            device=device,
        )
        map_residual_meas = get_residual_measurements(
            map_cat,
            image,
            paddings=padding,
            sources=map_reconstructions,
        )

        # finally get sep prediction, using BLISS for deblending
        sep_cat = get_sep_catalog(image, slen=slen, bp=bp)

        # now we get intermediate based on these locations so that we decide which locs
        # to keep in each tile, no deblending should be fine for this purpose
        # this could be technically done in the `get_sep_catalog` function
        _size = slen + 2 * bp
        _dummy_images = torch.zeros(1, sep_cat.max_n_sources, 1, _size, _size)
        sep_cat["fluxes"] = get_residual_measurements(
            sep_cat, image, paddings=padding, sources=_dummy_images
        )["flux"]
        sep_tile_cat = sep_cat.to_tile_params(tile_slen, ignore_extra_sources=True)
        sep_galaxy_bools = rearrange(sep_tile_cat.n_sources, "n nth ntw-> n nth ntw 1")
        sep_tile_cat["galaxy_bools"] = sep_galaxy_bools.float()
        sep_galaxy_params = deblender.variational_mode(
            image.to(device), sep_tile_cat.locs.to(device)
        ).to("cpu")
        sep_tile_cat["galaxy_params"] = sep_galaxy_params * sep_galaxy_bools
        sep_tile_cat = sep_tile_cat.to("cpu")
        sep_tile_cat.pop("fluxes")  # we don't need this anymore
        sep_cat = sep_tile_cat.to_full_params()
        sep_reconstructions = get_deblended_reconstructions(
            sep_cat,
            deblender._dec,
            slen=slen,
            bp=bp,
            device=device,
        )
        sep_residual_meas = get_residual_measurements(
            sep_cat,
            image,
            paddings=padding,
            sources=sep_reconstructions,
        )

        # skip if no sources found to avoid crashes
        if map_cat.n_sources.item() == 0:
            out["map_flux"] = torch.nan
            out["map_fluxerr"] = torch.nan
            out["n_sources_map"] = 0
            out["map_plocs"] = torch.tensor([])

        else:
            map_dist = torch.norm(
                map_cat.plocs[0] - torch.tensor([slen / 2, slen / 2]).reshape(1, 2), dim=-1
            )
            map_idx = torch.argmin(map_dist).item()
            out["n_sources_map"] = map_cat.n_sources.item()
            out["map_flux"] = map_residual_meas["flux"][:, map_idx, 0].item()
            out["map_fluxerr"] = map_residual_meas["fluxerr"][:, map_idx, 0].item()
            out["map_plocs"] = map_cat.plocs[0]

        if sep_cat.n_sources.item() == 0:
            out["sep_flux"] = torch.nan
            out["sep_fluxerr"] = torch.nan
            out["n_sources_sep"] = 0
            out["sep_plocs"] = torch.tensor([])

        else:
            sep_dist = torch.norm(
                sep_cat.plocs[0] - torch.tensor([slen / 2, slen / 2]).reshape(1, 2), dim=-1
            )
            sep_idx = torch.argmin(sep_dist).item()
            out["n_sources_sep"] = sep_cat.n_sources.item()
            out["sep_flux"] = sep_residual_meas["flux"][:, sep_idx, 0].item()
            out["sep_fluxerr"] = sep_residual_meas["fluxerr"][:, sep_idx, 0].item()
            out["sep_plocs"] = sep_cat.plocs[0]

        outs.append(out)

    return outs


def _get_diagnostic_figures(*, out_dir: Path, results: dict, tag_txt: str):
    # get relevant variables from outs
    outs = results["outs"]
    bld = results["bld"]
    true_snr = results["true_snr"]
    true_plocs = results["true_plocs"]
    true_n_sources = results["true_n_sources"]
    true_flux = results["true_flux"]
    images = results["images"]

    # easy figures
    # snr figure
    fig, ax = plt.subplots(figsize=(8, 6))
    _, bins, _ = ax.hist(
        true_snr[:, 0, 0].ravel().log10(),
        bins=51,
        color="C0",
        histtype="step",
        label="SNR of galaxy 1",
    )
    ax.set_xlabel("log10(SNR)")
    fig.savefig(out_dir / f"snr_histogram_central{tag_txt}.png")
    plt.close(fig)

    assert bld.ndim == 1

    # blendedness figure
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 0.5, 21)
    ax.hist(
        bld,
        bins=bins,
        color="C0",
        histtype="step",
        label="Blendedness of galaxy 1",
    )
    ax.set_xlabel("Blendedness")

    fig.savefig(out_dir / f"blendedness_histogram_central{tag_txt}.png")
    plt.close(fig)

    # now we make figures across all images using the output
    # we will make a big PDF, one page per image containing 4 plots
    pdf_path = out_dir / f"central_sim_results{tag_txt}.pdf"
    random_indices = np.random.choice(len(outs), size=min(len(outs), 1000), replace=False)
    with PdfPages(pdf_path) as pdf:
        for jj in tqdm(random_indices, desc="Generating figures"):
            out = outs[jj]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

            # blendedness as global title
            idx = out["idx"].item()
            blendedness = bld[idx].item()
            snr1 = true_snr[idx, 0].item()
            fig.suptitle(
                f"Blendedness: {blendedness:.4f}, \n SNR1: {snr1:.2f} \n Index: {jj}", fontsize=16
            )

            # Plot detection probability
            im = ax1.imshow(out["det_prob"], cmap="summer", origin="lower")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            ax1.set_title("Detection Probability")
            ax1.set_xlabel("Tile X Position")
            ax1.set_ylabel("Tile Y Position")
            # add text to each matrix cell
            for (i, j), val in np.ndenumerate(out["det_prob"]):
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)

            # Plot location samples in tile (x, y together)
            all_locs = out["nonzero_locs"]
            ax2.hist(
                all_locs.numpy(),
                bins=21,
                color="C0",
                alpha=0.7,
                histtype="step",
            )
            ax2.axvline(all_locs.median().item(), color="C1", linestyle="--", label="Median")
            ax2.axvline(all_locs.mean().item(), color="C2", linestyle="--", label="Mean")
            ax2.axvline(0.5, color="k", linestyle="--", label="True loc")
            ax2.set_title("Locations Histogram")
            ax2.legend()

            # Plot sample fluxes
            if not out["sample_fluxes"].isnan().all():
                try:
                    fluxes = out["sample_fluxes"]
                    map_flux = out["map_flux"]
                    sep_flux = out["sep_flux"]
                    _tflux = true_flux[idx, 0, 0].item()
                    _n_matched_samples = torch.sum(~torch.isnan(fluxes)).item()
                    ax3.set_title("# matched samples: " + str(_n_matched_samples))
                    ax3.hist(
                        fluxes.numpy(),
                        bins=21,
                        color="C0",
                        alpha=0.7,
                        histtype="step",
                    )
                    ax3.axvline(
                        fluxes.nanmean().item(), color="red", linestyle="--", label="Mean Flux"
                    )
                    ax3.axvline(map_flux, color="blue", linestyle="-.", label="Map Flux")
                    ax3.axvline(sep_flux, color="green", linestyle="-.", label="SEP Flux")
                    ax3.axvline(_tflux, color="k", linestyle="-", label="True Flux")
                    ax3.legend()
                except ValueError as e:
                    print(f"Error plotting fluxes for index {idx}: {e}")

            # shade error on mean
            is_nan = torch.isnan(fluxes)
            if (~is_nan).sum() > 1:
                err = torch.std(fluxes[~is_nan]).item()
                ax3.fill_between(
                    [fluxes.nanmean() - err, fluxes.nanmean() + err],
                    0,
                    ax3.get_ylim()[1],
                    color="red",
                    alpha=0.2,
                    label="Error on Mean",
                )

            # Plot number of sources sampled
            n_sources_samples = out["n_sources_samples"]
            n_sources_map = out["n_sources_map"]
            n_sources_sep = out["n_sources_sep"]
            n_sources = true_n_sources[idx].item()
            ax4.hist(
                n_sources_samples.numpy(),
                bins=np.arange(0, 10) - 0.5,
                color="C0",
                alpha=0.7,
                histtype="step",
            )
            ax4.axvline(n_sources_map, color="blue", linestyle="--", label="Map N Sources")
            ax4.axvline(n_sources, color="black", linestyle="--", label="True N Sources")
            ax4.axvline(n_sources_sep, color="green", linestyle="--", label="SEP N Sources")
            ax4.set_title("Number of Sources Sampled")
            ax4.legend()

            # also plot image
            ax5.imshow(images[idx].numpy().squeeze(), cmap="gray", origin="lower")
            ax5.set_title("Original Image")

            # plot image with samples plocs and MAP plocs
            assert torch.all(out["sample_plocs"][:, 1] > 0)
            assert torch.all(out["sample_plocs"][:, 0] > 0)
            sample_x = out["sample_plocs"][:, 1].numpy() + 24 - 0.5
            sample_y = out["sample_plocs"][:, 0].numpy() + 24 - 0.5
            ax6.imshow(images[idx].numpy().squeeze(), cmap="gray", origin="lower")
            ax6.scatter(
                sample_x, sample_y, color="red", s=20, alpha=0.2, label="Sampled Plocs", marker="x"
            )

            if out["map_plocs"].numel() > 0:
                ax6.scatter(
                    out["map_plocs"][:, 1] + 24 - 0.5,
                    out["map_plocs"][:, 0] + 24 - 0.5,
                    color="blue",
                    s=30,
                    alpha=1.0,
                    marker="+",
                    label="MAP Plocs",
                )

            # sep plocs
            if out["sep_plocs"].numel() > 0:
                ax6.scatter(
                    out["sep_plocs"][:, 1] + 24 - 0.5,
                    out["sep_plocs"][:, 0] + 24 - 0.5,
                    color="green",
                    s=30,
                    alpha=1.0,
                    marker="*",
                    label="SEP Plocs",
                )

            # true plocs
            _tplocs = true_plocs[idx].numpy()
            ax6.scatter(
                _tplocs[:, 1] + 24 - 0.5,
                _tplocs[:, 0] + 24 - 0.5,
                color="y",
                s=30,
                alpha=1.0,
                marker="o",
                facecolors="none",
                label="True Plocs",
            )
            ax6.legend()

            # save the figure to the PDF as a new page
            pdf.savefig(fig)
            plt.close(fig)


def _make_final_results_figures(*, out_dir: Path, rslts: dict) -> None:
    # need to sort things first!!!!
    sorted_indices = [out["idx"] for out in rslts["outs"]]
    true_fluxes = rslts["true_flux"][sorted_indices][:, 0, 0]
    bld = rslts["bld"][sorted_indices]
    true_snr = rslts["true_snr"][sorted_indices][:, 0, 0]

    samples_fluxes = torch.stack([out["sample_fluxes"] for out in rslts["outs"]])
    map_fluxes = torch.tensor([out["map_flux"] for out in rslts["outs"]])
    sep_fluxes = torch.tensor([out["sep_flux"] for out in rslts["outs"]])

    mask = ~torch.isnan(samples_fluxes).all(dim=1) & (true_snr > 0)  # only single galaxy snr < 0
    samples_fluxes = samples_fluxes[mask]
    map_fluxes = map_fluxes[mask]
    sep_fluxes = sep_fluxes[mask]
    true_fluxes = true_fluxes[mask]
    true_snr = true_snr[mask]
    bld = bld[mask]

    res1 = (samples_fluxes.nanmean(dim=1) - true_fluxes) / true_fluxes
    res2 = (map_fluxes - true_fluxes) / true_fluxes
    res3 = (sep_fluxes - true_fluxes) / true_fluxes

    stds = []
    for ii in range(len(samples_fluxes)):
        mask = ~torch.isnan(samples_fluxes[ii])
        if mask.sum() > 1:
            stds.append(torch.std(samples_fluxes[ii][mask]).item())
        else:
            stds.append(torch.nan)
    stds = torch.tensor(stds)

    z_score = (samples_fluxes.nanmean(dim=1) - true_fluxes) / stds

    mask_all = (
        ~torch.isnan(true_snr)
        & (true_snr > 0)
        & ~torch.isnan(bld)
        & ~torch.isnan(res1)
        & ~torch.isnan(res2)
        & ~torch.isnan(z_score)
        & ~torch.isnan(stds)
        & ~torch.isnan(sep_fluxes)
    )

    res1 = res1[mask_all]
    res2 = res2[mask_all]
    res3 = res3[mask_all]
    z_score = z_score[mask_all]
    bld = bld[mask_all]
    true_snr = true_snr[mask_all]
    stds = stds[mask_all]

    # get snr figure
    set_rc_params()

    # now snr
    n_bins = 21
    out1 = equal_sized_bin_statistic(
        x=true_snr.log10(), y=res1, n_bins=n_bins, xlims=(0.5, 3), statistic="median"
    )
    out2 = equal_sized_bin_statistic(
        x=true_snr.log10(), y=res2, n_bins=n_bins, xlims=(0.5, 3), statistic="median"
    )
    out3 = equal_sized_bin_statistic(
        x=true_snr.log10(), y=res3, n_bins=n_bins, xlims=(0.5, 3), statistic="median"
    )
    assert torch.all(out1["middles"] == out2["middles"])
    x = 10 ** out1["middles"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x, out3["stats"], label=r"\rm SEP", marker="", color=CLR_CYCLE[2])
    ax.fill_between(
        x,
        out3["stats"] - out3["errs"],
        out3["stats"] + out3["errs"],
        alpha=0.2,
        color=CLR_CYCLE[2],
    )
    ax.plot(x, out2["stats"], label=r"\rm MAP", marker="", color=CLR_CYCLE[0])
    ax.fill_between(
        x,
        out2["stats"] - out2["errs"],
        out2["stats"] + out2["errs"],
        alpha=0.2,
        color=CLR_CYCLE[0],
    )
    ax.plot(x, out1["stats"], label=r"\rm Samples", marker="", color=CLR_CYCLE[1])
    ax.fill_between(
        x,
        out1["stats"] - out1["errs"],
        out1["stats"] + out1["errs"],
        alpha=0.2,
        color=CLR_CYCLE[1],
    )
    ax.set_xlabel(r"\rm SNR", fontsize=28)
    ax.set_ylabel(r"$\frac{f_{\rm pred} - f_{\rm true}}{f_{\rm true}}$", fontsize=32)
    ax.axhline(0, color="k", linestyle="--", label=r"\rm Zero Residual")
    ax.legend()
    ax.set_xlim(5, 1000)
    ax.set_xscale("log")
    fig.savefig(out_dir / "samples_snr_res.png", dpi=500, bbox_inches="tight")

    # as a function of blendedness

    # first define bins (as described in paper)
    qs = torch.linspace(0.12, 0.99, 31)
    edges = bld.quantile(qs)
    bins = torch.tensor([0.0, *edges[1:-1], 1.0])

    out1 = binned_statistic(
        x=bld,
        y=res1,
        bins=bins,
        statistic="median",
    )
    out2 = binned_statistic(
        x=bld,
        y=res2,
        bins=bins,
        statistic="median",
    )
    out3 = binned_statistic(
        x=bld,
        y=res3,
        bins=bins,
        statistic="median",
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(out3["middles"], out3["stats"], label=r"\rm SEP", marker="", color=CLR_CYCLE[2])
    ax.fill_between(
        out3["middles"],
        out3["stats"] - out3["errs"],
        out3["stats"] + out3["errs"],
        alpha=0.2,
        color=CLR_CYCLE[2],
    )
    ax.plot(out2["middles"], out2["stats"], label=r"\rm MAP", marker="", color=CLR_CYCLE[0])
    ax.fill_between(
        out2["middles"],
        out2["stats"] - out2["errs"],
        out2["stats"] + out2["errs"],
        alpha=0.2,
        color=CLR_CYCLE[0],
    )
    ax.plot(out1["middles"], out1["stats"], label=r"\rm Samples", marker="", color=CLR_CYCLE[1])
    ax.fill_between(
        out1["middles"],
        out1["stats"] - out1["errs"],
        out1["stats"] + out1["errs"],
        alpha=0.2,
        color=CLR_CYCLE[1],
    )
    ax.set_xlabel(r"\rm Blendedness", fontsize=28)
    ax.set_ylabel(r"$\frac{f_{\rm pred} - f_{\rm true}}{f_{\rm true}}$", fontsize=32)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 10)
    ax.legend(prop={"size": 22})
    fig.savefig(out_dir / "samples_bld_res.png", dpi=500, bbox_inches="tight")

    print(
        f"Last blendedness bins comparisons: SEP {out3['stats'][-1]:.3f}, MAP {out2['stats'][-1]:.3f}, Samples {out1['stats'][-1]:.3f}"
    )


def main(
    seed: int = typer.Option(),
    n_images: int = 10_000,
    n_samples: int = 100,
    bp: int = 24,
    max_n_sources: int = 10,
    overwrite: bool = False,
    do_diagnostics: bool = False,
):
    pl.seed_everything(seed)

    device = torch.device("cuda:0")
    detection_fpath = MODELS_DIR / f"detection_{seed}.pt"
    ae_fpath = MODELS_DIR / f"autoencoder_{seed}.pt"
    deblend_fpath = MODELS_DIR / f"deblender_{seed}.pt"
    dataset_path = DATASETS_DIR / f"central_ds_{seed}.npz"
    results_path = CACHE_DIR / f"central_samples_results_{seed}.pt"

    assert dataset_path.exists()
    assert ae_fpath.exists()
    assert deblend_fpath.exists()
    assert detection_fpath.exists()

    if overwrite or not results_path.exists():
        print(f"Dataset already exists at {dataset_path}. Loading...")
        ds = load_dataset_npz(dataset_path)
        print("Dataset loaded successfully.")
        slen = ds["images"].shape[-1] - 2 * bp

        truth = FullCatalog(
            slen,
            slen,
            {
                "n_sources": ds["n_sources"],
                "plocs": ds["plocs"],
                "galaxy_bools": ds["galaxy_bools"],
            },
        )

        im1 = ds["uncentered_sources"]
        im2 = ds["uncentered_sources"].sum(dim=1)
        blendedness = get_blendedness(im1, im2)
        assert blendedness.shape == (n_images, max_n_sources)
        bld = blendedness[:, 0]  # only keep central galaxy for now
        assert bld.ndim == 1
        assert bld.shape == (n_images,)

        true_meas = get_residual_measurements(
            truth,
            ds["images"],
            paddings=ds["paddings"],
            sources=ds["uncentered_sources"],
            no_bar=False,
        )
        true_snr = true_meas["snr"]

        # lets get models
        detection = DetectionEncoder().to(device).eval()
        _ = detection.load_state_dict(
            torch.load(detection_fpath, map_location=device, weights_only=True)
        )
        detection = detection.requires_grad_(False).eval().to(device)

        deblender = GalaxyEncoder(ae_fpath)
        deblender.load_state_dict(torch.load(deblend_fpath, map_location=device, weights_only=True))
        deblender = deblender.requires_grad_(False).to(device).eval()

        # iterate over images in increasing order of blendedness of first source
        sorted_indices = np.argsort(bld)
        outs = _get_sample_results(
            sorted_indices=sorted_indices,
            n_samples=n_samples,
            images=ds["images"],
            paddings=ds["paddings"],
            slen=slen,
            tile_slen=5,
            bp=bp,
            detection=detection,
            deblender=deblender,
            device=device,
        )
        # save results
        torch.save(
            {
                "outs": outs,
                "bld": bld,
                "true_snr": true_snr,
                "true_flux": true_meas["flux"],
                "true_plocs": truth.plocs,
                "true_n_sources": truth.n_sources,
                "images": ds["images"],
            },
            results_path,
        )

    print(f"Results already exist at {results_path}. Loading...")
    results = torch.load(results_path, weights_only=False)
    print("Results loaded successfully.")
    print("Number of images:", len(results["outs"]))

    if do_diagnostics:
        _get_diagnostic_figures(FIGURE_DIR / str(seed), results, f"_{seed}")

    _make_final_results_figures(out_dir=FIGURE_DIR / str(seed), rslts=results)


if __name__ == "__main__":
    typer.run(main)
