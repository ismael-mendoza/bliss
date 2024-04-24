import numpy as np
import torch
from astropy.table import Table
from matplotlib.figure import Figure

from bliss.encoders.encoder import Encoder
from bliss.plotting import BlissFigure


class ToySeparationFigure(BlissFigure):
    @property
    def all_rcs(self):
        return {
            "three_separations": {
                "fontsize": 22,
                "tick_label_size": "small",
                "legend_fontsize": "small",
            },
            "toy_residuals": {"fontsize": 22},
            "toy_measurements": {
                "fontsize": 22,
                "tick_label_size": "small",
                "legend_fontsize": "small",
            },
        }

    @property
    def cache_name(self) -> str:
        return "toy_separation"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("three_separations", "toy_residuals", "toy_measurements")

    @property
    def separations_to_plot(self) -> list[int]:
        return [4, 8, 12]

    def compute_data(self, encoder: Encoder, decoder, galaxy_generator):
        # first, decide image size
        slen = 44
        bp = encoder.detection_encoder.bp
        tile_slen = encoder.detection_encoder.tile_slen
        size = 44 + 2 * bp
        tile_slen = encoder.detection_encoder.tile_slen
        assert slen / tile_slen % 2 == 1, "Need odd number of tiles to center galaxy."

        # now separations between galaxies to be considered (in pixels)
        # for efficiency, we set the batch_size equal to the number of separations
        seps = torch.arange(0, 18, 0.1)
        batch_size = len(seps)

        # Params: total_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a
        # first centered galaxy, then moving one.
        # BTK params: ra, dec, fluxnorm_disk, fluxnorm_bulge, fluxnorm_agn, a_d, b_d, a_b, b_b,
        # pa_bulge, pa_disk, btk_rotation
        colnames = [
            "ra",
            "dec",
            "fluxnorm_disk",
            "fluxnorm_bulge",
            "fluxnorm_agn",
            "a_d",
            "b_d",
            "a_b",
            "b_b",
            "pa_bulge",
            "pa_disk",
            "btk_rotation",
            "i_ab",
        ]
        flux1, flux2 = 2e5, 1e5
        mag1, mag2 = convert_flux_to_mag(torch.tensor([flux1, flux2]))
        mag1, mag2 = mag1.item(), mag2.item()
        gal1 = [0, 0, 1.0, 0, 0, 1.5, 0.7, 0, 0, np.pi / 4, np.pi / 4, 0, mag1]
        gal2 = [0, 0, 1.0, 0, 0, 1.0, 0.7, 0, 0, 3 * np.pi / 4, np.pi / 4, 0, mag2]
        assert len(gal1) == len(gal2) == len(colnames)
        base_cat = Table((gal1, gal2), names=colnames)

        # need plocs for later
        x0, y0 = 22, 22  # center plocs
        plocs = torch.tensor([[[x0, y0], [x0, y0 + sep]] for sep in seps]).reshape(batch_size, 2, 2)

        # setup btk configurations
        survey = btk.survey.get_surveys("LSST")
        i_band = survey.get_filter("i")
        psf = get_default_lsst_psf()
        bg = get_default_lsst_background()

        # create full catalogs (need separately since `render_blend`` only accepts 1 batch)
        images = torch.zeros(batch_size, 1, size, size)
        background = torch.zeros(batch_size, 1, size, size)
        for ii in range(batch_size):
            cat = base_cat.copy()
            ra = seps[ii] * PIXEL_SCALE
            cat["ra"] = ra
            image = _render_blend(cat, survey, i_band, psf, size)
            images[ii] = image
            background[ii] = bg

        # predictions from encoder
        tile_est = encoder.variational_mode(images, background)
        recon = decoder.render_images(tile_est)
        tile_est.set_all_fluxes_and_mags(decoder)
        tile_est = tile_est.cpu()
        recon = recon.detach().cpu() + background
        residuals = (recon - images) / recon.sqrt()

        # now we need to obtain flux, pred. ploc, prob. of detection in tile and std. of ploc
        # for each source
        params = {
            "images": images,
            "recon": recon,
            "resid": residuals,
            "seps": seps,
            "truth": {
                "flux": torch.tensor([flux1, flux2]).reshape(1, 2, 1).expand(batch_size, 2, 1),
                "ploc": plocs,
            },
            "est": {
                "prob_n_source": torch.zeros(batch_size, 2, 1),
                "flux": torch.zeros(batch_size, 2, 1),
                "ploc": torch.zeros(batch_size, 2, 2),
                "ploc_sd": torch.zeros(batch_size, 2, 2),
            },
            "tile_est": tile_est.to_dict(),
        }
        for ii, sep in enumerate(seps):
            # get tile_est for a single batch
            d = tile_est.to_dict()
            d = {k: v[ii, None] for k, v in d.items()}
            tile_est_ii = TileCatalog(tile_slen, d)

            ploc = plocs[ii]
            params_at_coord = tile_est_ii._get_tile_params_at_coord(ploc)
            prob_n_source = torch.exp(params_at_coord["n_source_log_probs"])
            flux = params_at_coord["fluxes"]
            ploc_sd = params_at_coord["loc_sd"] * tile_slen
            loc = params_at_coord["locs"]
            assert prob_n_source.shape == flux.shape == (2, 1)
            assert ploc_sd.shape == loc.shape == (2, 2)

            if sep < 2:
                params["est"]["prob_n_source"][ii][0] = prob_n_source[0]
                params["est"]["flux"][ii][0] = flux[0]
                params["est"]["ploc"][ii][0] = loc[0] * tile_slen + 5 * tile_slen
                params["est"]["ploc_sd"][ii][0] = ploc_sd[0]

                params["est"]["prob_n_source"][ii][1] = torch.nan
                params["est"]["flux"][ii][1] = torch.nan
                params["est"]["ploc"][ii][1] = torch.tensor([torch.nan, torch.nan])
                params["est"]["ploc_sd"][ii][1] = torch.tensor([torch.nan, torch.nan])
            else:
                bias = 5 + np.ceil((sep - 2) / 4)
                params["est"]["prob_n_source"][ii] = prob_n_source
                params["est"]["flux"][ii] = flux
                params["est"]["ploc"][ii][0] = loc[0] * tile_slen + 5 * tile_slen
                params["est"]["ploc"][ii, 1, 0] = loc[1][0] * tile_slen + 5 * tile_slen
                params["est"]["ploc"][ii, 1, 1] = loc[1][1] * tile_slen + bias * tile_slen
                params["est"]["ploc_sd"][ii] = ploc_sd

        return params

    def _get_three_separations_plot(self, data) -> Figure:
        seps: np.ndarray = data["seps"]
        images: np.ndarray = data["images"]
        tplocs: np.ndarray = data["truth"]["ploc"]
        eplocs: np.ndarray = data["est"]["ploc"]

        # first, create image with 3 example separations (very blended to not blended)
        bp = 24
        fig, axes = plt.subplots(1, 3, figsize=(12, 7))
        axes = axes.flatten()
        seps_to_plot = self.separations_to_plot
        trim = 25  # zoom into relevant part of the image

        c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]  # true
        c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]  # predicted

        for ii, psep in enumerate(seps_to_plot):
            indx = list(seps).index(psep)
            image = images[indx, 0, trim:-trim, trim:-trim]
            x1 = tplocs[indx, :, 1] + bp - 0.5 - trim
            y1 = tplocs[indx, :, 0] + bp - 0.5 - trim
            x2 = eplocs[indx, :, 1] + bp - 0.5 - trim
            y2 = eplocs[indx, :, 0] + bp - 0.5 - trim
            axes[ii].imshow(image, cmap="gray")
            axes[ii].scatter(x1, y1, marker="x", color="r", s=30, label=None if ii else "Truth")
            axes[ii].scatter(x2, y2, marker="+", color="b", s=50, label=None if ii else "Predicted")
            axes[ii].set_xticks([0, 10, 20, 30, 40])
            axes[ii].set_yticks([0, 10, 20, 30, 40])
            axes[ii].set_title(rf"\rm Separation: {psep} pixels")

            if ii == 0:
                axes[ii].legend(loc="best", prop={"size": 14}, markerscale=2)

            if ii > 0:
                axes[ii].set_yticks([])  # turn off axis
                axes[ii].set_ylim(axes[0].get_ylim())  # align axes

            axes[ii].text(x1[0].item(), y1[0].item() - 7, "1", color=c1)
            axes[ii].text(x1[1].item(), y1[1].item() - 7, "2", color=c2)

        fig.tight_layout()

        return fig

    def _get_residuals_figure(self, data) -> Figure:
        n_examples = 3
        bp = 24
        seps_to_plot = [4, 8, 12]
        seps = data["seps"]
        tplocs: np.ndarray = data["truth"]["ploc"]
        images_all, recon_all, res_all = data["images"], data["recon"], data["resid"]
        trim = 20
        indices = np.array([list(seps).index(psep) for psep in seps_to_plot]).astype(int)

        images = images_all[indices, 0, trim + 10 : -trim - 10, trim + 15 : -trim - 5]
        recons = recon_all[indices, 0, trim + 10 : -trim - 10, trim + 15 : -trim - 5]
        residuals = res_all[indices, 0, trim + 10 : -trim - 10, trim + 15 : -trim - 5]
        x1 = tplocs[indices, :, 1] + bp - 0.5 - trim - 15
        y1 = tplocs[indices, :, 0] + bp - 0.5 - trim - 10

        pad = 6.0
        fig, axes = plt.subplots(nrows=n_examples, ncols=3, figsize=(11, 18))

        for i in range(n_examples):
            ax_true = axes[i, 0]
            ax_recon = axes[i, 1]
            ax_res = axes[i, 2]

            # only add titles to the first axes.
            if i == 0:
                ax_true.set_title(r"\rm Images $x$", pad=pad)
                ax_recon.set_title(r"\rm Reconstruction $\tilde{x}$", pad=pad)
                ax_res.set_title(
                    r"Residual $\left(\tilde{x} - x\right) / \sqrt{\tilde{x}}$", pad=pad
                )

            ax_true.scatter(x1[i], y1[i], color="r", alpha=0.5, s=40, marker="x", label="Truth")
            ax_recon.scatter(x1[i], y1[i], color="r", alpha=0.5, s=40, marker="x")
            ax_res.scatter(x1[i], y1[i], color="r", alpha=0.5, s=40, marker="x")

            # standarize ranges of true and reconstruction
            image = images[i]
            recon = recons[i]
            res = residuals[i]

            vmin = min(image.min().item(), recon.min().item())
            vmax = max(image.max().item(), recon.max().item())
            vmin_res = res.min().item()
            vmax_res = res.max().item()

            # plot images
            plot_image(fig, ax_true, image, vrange=(vmin, vmax))
            plot_image(fig, ax_recon, recon, vrange=(vmin, vmax))
            plot_image(fig, ax_res, res, vrange=(vmin_res, vmax_res))

            ax_true.set_xticks([0, 10, 20, 30])
            ax_true.set_yticks([0, 10, 20, 30])
            ax_recon.set_xticks([0, 10, 20, 30])
            ax_recon.set_yticks([0, 10, 20, 30])
            ax_res.set_xticks([0, 10, 20, 30])
            ax_res.set_yticks([0, 10, 20, 30])

            if i == 0:
                ax_true.legend(loc="best", prop={"size": 14}, markerscale=2)

        plt.subplots_adjust(hspace=-0.9)
        plt.tight_layout()
        return fig

    def _get_measurement_figure(self, data: dict) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axs = axes.flatten()
        seps = data["seps"]
        xticks = [sep for sep in seps if sep % 2 == 0]

        c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
        c2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]

        # probability of detection in each tile
        prob_n1 = data["est"]["prob_n_source"][:, 0]
        prob_n2 = data["est"]["prob_n_source"][:, 1]

        axs[0].plot(seps, prob_n1, "-", label="Galaxy 1", color=c1)
        axs[0].plot(seps, prob_n2, "-", label="Galaxy 2", color=c2)
        axs[0].axvline(2, color="k", ls="--", label=r"\rm Tile boundary", alpha=0.5)
        axs[0].axvline(6, ls="--", color="k", alpha=0.5)
        axs[0].axvline(10, ls="--", color="k", alpha=0.5)
        axs[0].axvline(14, ls="--", color="k", alpha=0.5)
        axs[0].legend(loc="best")
        axs[0].set_xticks(xticks)
        axs[0].set_xlim(0, 16)
        axs[0].set_xlabel(r"\rm Separation (pixels)")
        axs[0].set_ylabel(r"\rm Detection Probability")

        # distance residual
        tploc1 = data["truth"]["ploc"][:, 0]
        tploc2 = data["truth"]["ploc"][:, 1]
        eploc1 = data["est"]["ploc"][:, 0]
        eploc2 = data["est"]["ploc"][:, 1]
        dist1 = ((tploc1 - eploc1) ** 2).sum(1) ** (1 / 2)
        dist2 = ((tploc2 - eploc2) ** 2).sum(1) ** (1 / 2)

        axs[1].plot(seps, dist1, "-", color=c1)
        axs[1].plot(seps, dist2, "-", color=c2)
        axs[1].axhline(0, color="k", ls="-", alpha=1.0)
        axs[1].axvline(2, ls="--", color="k", label=r"\rm Tile boundary", alpha=0.5)
        axs[1].axvline(6, ls="--", color="k", alpha=0.5)
        axs[1].axvline(10, ls="--", color="k", alpha=0.5)
        axs[1].axvline(14, ls="--", color="k", alpha=0.5)
        axs[1].set_xticks(xticks)
        axs[1].set_xlim(0, 16)
        axs[1].set_xlabel(r"\rm Separation (pixels)")
        axs[1].set_ylabel(r"\rm Centroid location residual (pixels)")

        # location error (squared sum) estimate
        eploc_sd1 = (data["est"]["ploc_sd"][:, 0] ** 2).sum(1) ** (1 / 2)
        eploc_sd2 = (data["est"]["ploc_sd"][:, 1] ** 2).sum(1) ** (1 / 2)
        axs[3].plot(seps, eploc_sd1, "-", color=c1)
        axs[3].plot(seps, eploc_sd2, "-", color=c2)
        axs[3].axhline(0, color="k", ls="-", alpha=1.0)
        axs[3].axvline(2, ls="--", color="k", label=r"\rm Tile boundary", alpha=0.5)
        axs[3].axvline(6, ls="--", color="k", alpha=0.5)
        axs[3].axvline(10, ls="--", color="k", alpha=0.5)
        axs[3].axvline(14, ls="--", color="k", alpha=0.5)
        axs[3].set_xticks(xticks)
        axs[3].set_xlim(0, 16)
        axs[3].set_ylim(axs[1].get_ylim())
        axs[3].set_xlabel(r"\rm Separation (pixels)")
        axs[3].set_ylabel(r"\rm Predicted centroid std. (pixels)")

        # flux normalized residuals
        tflux1 = data["truth"]["flux"][:, 0]
        tflux2 = data["truth"]["flux"][:, 1]
        eflux1 = data["est"]["flux"][:, 0]
        eflux2 = data["est"]["flux"][:, 1]
        rflux1 = (eflux1 - tflux1) / tflux1
        rflux2 = (eflux2 - tflux2) / tflux2

        axs[2].plot(seps, rflux1, "-", color=c1)
        axs[2].plot(seps, rflux2, "-", color=c2)
        axs[2].axhline(0, color="k", ls="-", alpha=1.0)
        axs[2].axvline(2, ls="--", color="k", label=r"\rm Tile boundary", alpha=0.5)
        axs[2].axvline(6, ls="--", color="k", alpha=0.5)
        axs[2].axvline(10, ls="--", color="k", alpha=0.5)
        axs[2].axvline(14, ls="--", color="k", alpha=0.5)
        axs[2].set_xticks(xticks)
        axs[2].set_xlim(0, 16)
        axs[2].set_xlabel(r"\rm Separation (pixels)")
        axs[2].set_ylabel(r"\rm $(f^{\rm recon} - f^{\rm true}) / f^{\rm true}$")

        return fig

    def create_figure(self, fname: str, data: dict) -> Figure:
        if fname == "three_separations":
            return self._get_three_separations_plot(data)
        if fname == "toy_residuals":
            return self._get_residuals_figure(data)
        if fname == "toy_measurements":
            return self._get_measurement_figure(data)
        raise NotImplementedError("Figure {fname} not implemented.")


def _render_blend(blend_cat: Table, survey, filt, psf, slen: int):
    blend_image = torch.zeros((1, slen, slen))
    for entry in blend_cat:
        single_image = btk.draw_blends.render_single_catsim_galaxy(entry, filt, survey, psf, slen)
        blend_image[0] += single_image.array
    return blend_image
