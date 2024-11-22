import torch

from bliss.catalog import FullCatalog
from bliss.reporting import compute_batch_tp_fp, match_and_classify


def test_metrics():
    slen = 50
    slack = 1.0

    true_locs = torch.tensor(
        [[[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]], [[0.1, 0.1], [0.2, 0.2], [0.0, 0.0]]]
    ).reshape(2, 3, 2)
    est_locs = torch.tensor(
        [[[0.49, 0.49], [0.1, 0.1], [0.0, 0.0]], [[0.19, 0.19], [0.01, 0.01], [0.0, 0.0]]]
    ).reshape(2, 3, 2)
    true_galaxy_bools = torch.tensor([[1, 0, 0], [1, 1, 0]]).reshape(2, 3, 1)
    est_galaxy_bools = torch.tensor([[0, 1, 0], [1, 0, 0]]).reshape(2, 3, 1)
    true_star_bools = torch.tensor([[0, 0, 0], [0, 0, 0]]).reshape(2, 3, 1)
    est_star_bools = torch.tensor([[1, 0, 0], [0, 1, 0]]).reshape(2, 3, 1)

    true_params = FullCatalog(
        slen,
        slen,
        {
            "n_sources": torch.tensor([1, 2]),
            "plocs": true_locs * slen,
            "galaxy_bools": true_galaxy_bools,
            "star_bools": true_star_bools,
        },
    )
    est_params = FullCatalog(
        slen,
        slen,
        {
            "n_sources": torch.tensor([2, 2]),
            "plocs": est_locs * slen,
            "galaxy_bools": est_galaxy_bools,
            "star_bools": est_star_bools,
        },
    )

    tp, fp, ntrue = compute_batch_tp_fp(true_params, est_params)
    precision = tp.sum() / (tp.sum() + fp.sum())
    recall = tp.sum() / ntrue.sum()

    results_classify = match_and_classify(true_params, est_params)
    tp_gal = results_classify["tp_gal"]
    tp_star = results_classify["tp_star"]
    n_matches = results_classify["n_gal"] + results_classify["n_star"]
    class_acc = (tp_gal.sum() + tp_star.sum()) / n_matches.sum()

    assert precision == 2 / (2 + 2)
    assert recall == 2 / 3
    assert class_acc == 1 / 2
