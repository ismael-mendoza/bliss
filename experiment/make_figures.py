import subprocess

import pytorch_lightning as L
import typer

from experiment import CACHE_DIR, FIGURE_DIR, SEED


def main(
    detection: bool = False,
    binary: bool = False,
    deblend: bool = False,
    toy: bool = False,
    samples: bool = False,
    all: bool = False,
    overwrite: bool = False,
):
    L.seed_everything(SEED)
    FIGURE_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    if detection or all:
        cmd = f"./scripts/get_figures.py --mode detection --overwrite {overwrite}"
        subprocess.check_call(cmd, shell=True)

    if binary or all:
        cmd = f"./scripts/get_figures.py --mode binary --overwrite {overwrite}"
        subprocess.check_call(cmd, shell=True)

    if deblend or all:
        cmd = f"./scripts/get_figures.py --mode deblend --overwrite {overwrite}"
        subprocess.check_call(cmd, shell=True)

    if toy or all:
        cmd = f"./scripts/get_figures.py --mode toy --overwrite {overwrite}"
        subprocess.check_call(cmd, shell=True)

    if samples or all:
        cmd = f"./scripts/get_figures.py --mode samples --overwrite {overwrite}"
        subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    typer.run(main)
