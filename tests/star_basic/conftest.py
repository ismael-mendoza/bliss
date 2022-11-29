from pathlib import Path

import pytest
from hydra import compose, initialize

from tests.conftest import ModelSetup


def get_star_basic_cfg(overrides):
    overrides.update(
        {"training.weight_save_path": None, "paths.root": Path(__file__).parents[2].as_posix()}
    )
    overrides = [f"{k}={v}" if v is not None else f"{k}=null" for k, v in overrides.items()]
    with initialize(config_path=".", version_base=None):
        cfg = compose("star_basic", overrides=overrides)
    return cfg


class StarBasicModelSetup(ModelSetup):
    def get_cfg(self, overrides):
        return get_star_basic_cfg(overrides)


@pytest.fixture(scope="session")
def star_basic_model_setup(devices):
    return StarBasicModelSetup(devices)
