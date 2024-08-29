from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from einops import reduce

from bliss.datasets.galsim_blends import generate_individual_dataset
from bliss.datasets.lsst import convert_mag_to_flux, get_default_lsst_psf
from bliss.reporting import get_single_galaxy_ellipticities, get_snr


def test_galaxy_blend_catalogs(home_dir: Path):
    psf = get_default_lsst_psf()
    catsim_table = Table.read(home_dir / "data" / "catsim_snr.fits")
    snr_mask = catsim_table["snr"] > 10
    final_table = catsim_table[snr_mask]

    ds = generate_individual_dataset(3000, final_table, psf, slen=53, replace=False)

    image_fluxes = reduce(ds["noiseless"], "b c h w -> b", "sum")
    saved_fluxes = ds["galaxy_params"][:, -1]
    mags = ds["galaxy_params"][:, -2]
    adjusted_fluxes = []  # ignore agn contribution

    # ignore agn contribution
    for ii in range(3000):
        fnb, fnd, fnagn, _, _, _, _, _, _, _, total_flux = ds["galaxy_params"][ii]
        adjusted_fluxes.append(total_flux / (fnb + fnd + fnagn) * (fnb + fnd))
    adjusted_fluxes = torch.tensor(adjusted_fluxes)
    cat_fluxes = convert_mag_to_flux(mags)

    # check indices match
    assert np.all(np.isin(ds["indices"].numpy(), final_table["galtileid"]))

    # check fluxes match
    assert np.allclose(cat_fluxes, saved_fluxes, rtol=1e-4, atol=0)
    res = (adjusted_fluxes - image_fluxes) / adjusted_fluxes
    assert np.all(res.numpy() > 0)
    assert sum(res.numpy() > 0.1) / 3000 < 0.01  # not too many big galaxies

    # check snr cut was correctly propagated to images
    image_snr = get_snr(ds["noiseless"], ds["background"]).numpy()
    assert np.all(image_snr > 9.9)  # fudge factor

    # check ellpiticity distribution is correct
    ellips = get_single_galaxy_ellipticities(ds["noiseless"][:, 0, :, :])
    e1 = ellips[:, 0].numpy()
    e2 = ellips[:, 1].numpy()
    mask = ~np.isnan(e1)
    e1 = e1[mask]
    e2 = e2[mask]

    np.allclose(np.mean(e1), 0.0, rtol=0.0, atol=1e-2)
    np.allclose(np.mean(e2), 0.0, rtol=0.0, atol=1e-2)

    np.allclose(np.std(e1), 0.15, rtol=0.0, atol=1e-2)
    np.allclose(np.std(e2), 0.15, rtol=0.0, atol=1e-2)
