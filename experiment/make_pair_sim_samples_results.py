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
from bliss.datasets.lsst import get_default_lsst_psf, prepare_final_galaxy_catalog
from bliss.datasets.pair_sim import generate_pair_dataset
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.reporting import (
    get_blendedness,
    get_deblended_reconstructions,
    get_residual_measurements,
)


def get_sample_results(
    *,
    sorted_indices,
    n_samples: int,
    detection: DetectionEncoder,
    deblender: GalaxyEncoder,
    images: torch.Tensor,
    device: torch.device,
) -> list[dict]:
    outs = []
    for ii in tqdm(sorted_indices, desc="Processing images"):
        out = {}
        image = images[ii, None]

        det_prob, _, _ = detection.forward(image.to(device))
        det_prob = det_prob.cpu()

        samples = detection.sample(image.to(device), n_samples=n_samples)
        samples = {k: v.to("cpu") for k, v in samples.items()}

        # look at locs
        _locs = samples["locs"][:, 25 // 2, :].flatten()
        nonzero_locs = _locs[_locs.nonzero()]
        out["nonzero_locs"] = nonzero_locs

        # now get cats
        # this function atuomatically zeroes out sources which locs out of tile
        tile_cats = turn_samples_into_catalogs(samples, tile_slen=5, nth=5, ntw=5)

        # add galaxy params to each catalog
        for jj in tqdm(range(n_samples), desc="Adding galaxy params to catalogs", disable=True):
            galaxy_bools = rearrange(tile_cats[jj].n_sources, "n nth ntw-> n nth ntw 1")
            tile_cats[jj]["galaxy_bools"] = galaxy_bools.float()
            galaxy_params = deblender.variational_mode(
                image.to(device), tile_cats[jj].locs.to(device)
            ).to("cpu")
            galaxy_params *= tile_cats[jj]["galaxy_bools"]
            tile_cats[jj]["galaxy_params"] = galaxy_params

        sample_cats = []
        for kk in range(len(tile_cats)):
            sample_cats.append(tile_cats[kk].to_full_params())

        reconstructions = []
        for rr in tqdm(range(n_samples), desc="Reconstructing samples", disable=True):
            recon_uncentered = get_deblended_reconstructions(
                sample_cats[rr],
                deblender._dec,
                slen=25,
                device=device,
            )
            reconstructions.append(recon_uncentered)

        residual_meas = []
        for ll in tqdm(range(n_samples), desc="Calculating residual measurements", disable=True):
            meas = get_residual_measurements(
                sample_cats[ll],
                image,
                paddings=torch.zeros_like(image),  # we don't have padding only one more galaxy
                sources=reconstructions[ll],
            )
            assert meas["flux"].shape[0] == 1
            residual_meas.append(meas)

        # pick fluxes that are within central tile only (i.e. that match with central galaxy)
        sample_fluxes = []
        for ss in range(n_samples):
            meas = residual_meas[ss]
            _plocs = sample_cats[ss].plocs
            assert _plocs.shape[0] == 1
            plocs = _plocs[0]
            central_plocs = torch.tensor([12.5, 12.5]).reshape(1, 2)
            dist_to_center = torch.norm(plocs - central_plocs, dim=-1)
            indices = torch.argwhere(dist_to_center < 2).flatten()
            if len(indices) > 1:
                raise ValueError("More than one source within central tile found.")
            elif len(indices) == 0:
                sample_fluxes.append(torch.nan)
            else:
                _idx = indices.item()
                sample_fluxes.append(meas["flux"][0, _idx, 0].item())
        sample_fluxes = torch.tensor(sample_fluxes)

        # get map prediction too
        map_tile_cat = detection.variational_mode(image.to(device))
        map_tile_cat["galaxy_bools"] = rearrange(
            map_tile_cat.n_sources, "n nth ntw-> n nth ntw 1"
        ).float()
        map_galaxy_params = deblender.variational_mode(
            image.to(device), map_tile_cat.locs.to(device)
        )
        map_tile_cat["galaxy_params"] = map_galaxy_params
        map_tile_cat = map_tile_cat.to("cpu")

        map_cat = map_tile_cat.to_full_params()
        map_reconstructions = get_deblended_reconstructions(
            map_cat,
            deblender._dec,
            slen=25,
            device=device,
        )
        map_residual_meas = get_residual_measurements(
            map_cat,
            image,
            paddings=torch.zeros_like(image),
            sources=map_reconstructions,
        )

        map_idx = torch.argmin(
            torch.norm(map_cat.plocs[0] - torch.tensor([12.5, 12.5]).reshape(1, 2), dim=-1)
        ).item()
        map_flux = map_residual_meas["flux"][:, map_idx, 0].item()

        out["sample_fluxes"] = sample_fluxes
        out["map_flux"] = map_flux

        n_sources_samples = torch.tensor([cat.n_sources.item() for cat in sample_cats])
        out["n_sources_samples"] = n_sources_samples
        out["n_sources_map"] = map_cat.n_sources.item()
        out["det_prob"] = det_prob.reshape(5, 5).cpu()
        out["idx"] = ii

        # locations
        map_plocs = map_cat.plocs[0]
        sample_plocs = []
        for ss in range(n_samples):
            _plocs = sample_cats[ss].plocs[0]
            assert _plocs.shape[0] == sample_cats[ss].n_sources.item()  # only adding nonzero
            sample_plocs.append(_plocs)
        sample_plocs = torch.concatenate(sample_plocs, dim=0)

        out["sample_plocs"] = sample_plocs
        out["map_plocs"] = map_plocs

        outs.append(out)

    return outs


def main(n_images: int = 100, n_samples: int = 500, overwrite: bool = False):
    device = torch.device("cuda:0")
    out_dir = Path("figures/pair_sim")
    deblend_fpath = "models/deblender_23_22.pt"
    ae_fpath = "models/autoencoder_42_42.pt"
    results_path = out_dir / "pair_sim_results.pt"

    pl.seed_everything(42)

    cat = prepare_final_galaxy_catalog()
    psf = get_default_lsst_psf()
    _high_mag_cat = cat[cat["i_ab"] < 25.3]

    ds = generate_pair_dataset(
        n_images,
        _high_mag_cat,
        psf,
        out_square=15.0,
    )
    truth = FullCatalog(
        25,
        25,
        {"n_sources": ds["n_sources"], "plocs": ds["plocs"], "galaxy_bools": ds["galaxy_bools"]},
    )

    true_meas = get_residual_measurements(
        truth,
        ds["images"],
        paddings=torch.zeros_like(ds["images"]),
        sources=ds["uncentered_sources"],
    )

    # snr figure
    fig, ax = plt.subplots(figsize=(8, 6))
    _, bins, _ = ax.hist(
        true_meas["snr"][:, 0, 0].ravel().log10(),
        bins=51,
        color="C0",
        histtype="step",
        label="SNR of galaxy 1",
    )
    ax.hist(
        true_meas["snr"][:, 1, 0].ravel().log10(),
        bins=bins,
        color="C1",
        histtype="step",
        label="SNR of galaxy 2",
    )
    ax.set_xlabel("log10(SNR)")
    fig.savefig(out_dir / "snr_histogram.png")
    plt.close(fig)

    im1 = ds["uncentered_sources"]
    im2 = ds["uncentered_sources"].sum(dim=1)
    bld = get_blendedness(im1, im2)
    assert bld.shape == (n_images, 2)

    # blendedness figure
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 0.5, 21)
    ax.hist(
        bld[:, 0].ravel(),
        bins=bins,
        color="C0",
        histtype="step",
        label="Blendedness of galaxy 1",
    )
    ax.hist(
        bld[:, 1].ravel(),
        bins=bins,
        color="C1",
        histtype="step",
        label="Blendedness of galaxy 2",
    )
    ax.set_xlabel("Blendedness")

    fig.savefig(out_dir / "blendedness_histogram.png")
    plt.close(fig)

    # lets get models
    detection = DetectionEncoder().to(device).eval()
    _ = detection.load_state_dict(
        torch.load("models/detection_23_23.pt", map_location=device, weights_only=True)
    )
    detection = detection.requires_grad_(False).eval().to(device)

    deblender = GalaxyEncoder(ae_fpath)
    deblender.load_state_dict(torch.load(deblend_fpath, map_location=device, weights_only=True))
    deblender = deblender.requires_grad_(False).to(device).eval()

    # next part
    # iterate over images in increasing order of blendedness of first source
    sorted_indices = np.argsort(bld[:, 0].ravel())

    if results_path.exists() and not overwrite:
        print(f"Results already exist at {results_path}. Loading...")
        outs = torch.load(results_path)
        print("Results loaded successfully.")

    else:
        outs = get_sample_results(
            sorted_indices=sorted_indices,
            n_samples=n_samples,
            detection=detection,
            deblender=deblender,
            images=ds["images"],
            device=device,
        )
        # save results
        torch.save(outs, results_path)

    # now we make figures across all images using the output
    # we will make a big PDF, one page per image containing 4 plots
    pdf_path = out_dir / "pair_sim_results.pdf"
    with PdfPages(pdf_path) as pdf:
        for out in outs:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

            # blendedness as global title
            idx = out["idx"]
            blendedness = bld[idx, 0].item()
            fig.suptitle(f"Blendedness: {blendedness:.4f}", fontsize=16)

            # Plot detection probability
            im = ax1.imshow(out["det_prob"], cmap="viridis", origin="lower")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            ax1.set_title("Detection Probability")
            ax1.set_xlabel("Tile X Position")
            ax1.set_ylabel("Tile Y Position")

            # Plot location samples in tile (x, y together)
            all_locs = out["nonzero_locs"]
            ax2.hist(
                all_locs.numpy(),
                bins=50,
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
            idx = out["idx"]
            fluxes = out["sample_fluxes"]
            map_flux = out["map_flux"]
            true_flux = true_meas["flux"][idx, 0, 0].item()
            ax3.hist(
                fluxes.numpy(),
                bins=31,
                color="C0",
                alpha=0.7,
                histtype="step",
            )
            ax3.axvline(fluxes.nanmean().item(), color="red", linestyle="--", label="Mean Flux")
            ax3.axvline(true_flux, color="k", linestyle="--", label="True Flux")
            ax3.axvline(map_flux, color="blue", linestyle="--", label="Map Flux")
            ax3.legend()

            # shade error on mean
            is_nan = torch.isnan(fluxes)
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
            ax4.hist(
                n_sources_samples.numpy(),
                bins=np.arange(0, 10) - 0.5,
                color="C0",
                alpha=0.7,
                histtype="step",
            )
            ax4.axvline(n_sources_map, color="blue", linestyle="--", label="Map N Sources")
            ax4.set_title("Number of Sources Sampled")
            ax4.legend()

            # also plot image
            ax5.imshow(ds["images"][out["idx"]].numpy().squeeze(), cmap="gray", origin="lower")
            ax5.set_title("Original Image")

            # plot image with samples plocs and MAP plocs
            sample_x = out["sample_plocs"][:, 1].numpy() + 24 - 0.5
            sample_y = out["sample_plocs"][:, 0].numpy() + 24 - 0.5
            ax6.imshow(ds["images"][out["idx"]].numpy().squeeze(), cmap="gray", origin="lower")
            ax6.scatter(
                sample_x, sample_y, color="red", s=20, alpha=0.2, label="Sampled Plocs", marker="x"
            )
            ax6.scatter(
                out["map_plocs"][:, 1] + 24 - 0.5,
                out["map_plocs"][:, 0] + 24 - 0.5,
                color="blue",
                s=30,
                alpha=1.0,
                marker="+",
                label="MAP Plocs",
            )
            # true plocs
            true_plocs = truth.plocs[out["idx"]].numpy().squeeze()
            ax6.scatter(
                true_plocs[:, 1] + 24 - 0.5,
                true_plocs[:, 0] + 24 - 0.5,
                color="k",
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


if __name__ == "__main__":
    typer.run(main)
