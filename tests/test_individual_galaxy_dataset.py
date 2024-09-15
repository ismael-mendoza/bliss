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
    final_table = Table.read(home_dir / "data" / "OneDegSq.fits")

    n_gals = 1000
    ds = generate_individual_dataset(n_gals, final_table, psf, slen=53, replace=False)

    image_fluxes = reduce(ds["noiseless"], "b c h w -> b", "sum")
    saved_fluxes = ds["galaxy_params"][:, -1]
    mags = ds["galaxy_params"][:, -2]
    adjusted_fluxes = []  # ignore agn contribution

    # ignore agn contribution
    for ii in range(n_gals):
        fnb, fnd, fnagn, _, _, _, _, _, _, _, total_flux = ds["galaxy_params"][ii]
        adjusted_fluxes.append(total_flux / (fnb + fnd + fnagn) * (fnb + fnd))
    adjusted_fluxes = torch.tensor(adjusted_fluxes)
    cat_fluxes = convert_mag_to_flux(mags).float()  # float 32

    # check indices match
    ids = ds["indices"].numpy()
    cat_ids = final_table["galtileid"].value
    assert np.all(np.isin(ids, cat_ids)), "All ids should be contained in OG catalog"

    # check fluxes match
    assert np.allclose(cat_fluxes, saved_fluxes, rtol=1e-4, atol=0)
    res = (adjusted_fluxes - image_fluxes) / adjusted_fluxes
    assert np.all(res.numpy() > 0)
    assert sum(res.numpy() > 0.1) / 3000 < 0.01  # small fraction of big galaxies

    # check snr matches what is in the catalog to 1%
    mask = np.isin(cat_ids, ids)
    cat1 = final_table[mask]
    cat1.sort(keys="galtileid")  # sort in place
    assert np.all(cat1["galtileid"].value == np.sort(cat1["galtileid"].value))
    assert np.all(np.isin(cat1["galtileid"], ids))

    # check ellpiticity distribution is correct
    ellips = get_single_galaxy_ellipticities(ds["noiseless"][:, 0, :, :])
    e1 = ellips[:, 0].numpy()
    e2 = ellips[:, 1].numpy()
    mask = ~np.isnan(e1)
    e1 = e1[mask]
    e2 = e2[mask]

    # symmetric
    np.allclose(np.mean(e1), 0.0, rtol=0.0, atol=1e-2)
    np.allclose(np.mean(e2), 0.0, rtol=0.0, atol=1e-2)

    # reasonable scatter
    np.allclose(np.std(e1), 0.15, rtol=0.0, atol=1e-2)
    np.allclose(np.std(e2), 0.15, rtol=0.0, atol=1e-2)
