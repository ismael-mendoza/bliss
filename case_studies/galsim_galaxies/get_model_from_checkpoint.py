#!/usr/bin/env python3

from pathlib import Path

import click
import numpy as np
import torch


def _find_best_checkpoint(checkpoint_dir: str):
    """Given directory to checkpoints, automatically return file path to lowest loss checkpoint."""
    best_path = Path(".")
    min_loss = np.inf
    for pth in Path(checkpoint_dir).iterdir():
        if pth.stem.startswith("epoch"):
            # extract loss
            idx = pth.stem.find("=", len("epoch") + 1)
            loss = float(pth.stem[idx + 1 :])
            if loss < min_loss:
                best_path = pth
                min_loss = loss
    return best_path


def _save_weights(weight_save_path: str, model_checkpoint_path: str):
    model_checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model_state_dict = model_checkpoint["state_dict"]
    weight_file_path = Path(weight_save_path)
    assert weight_file_path.parent.exists()
    assert not weight_file_path.exists()
    torch.save(model_state_dict, weight_save_path)


@click.command()
@click.option("-w", "--weight-path", type=str, required=True)
@click.option("-c", "--checkpoint-dir", type=str, required=True)
def main(weight_path: str, checkpoint_dir: str):
    """Save weights from model checkpoint."""
    checkpoint_path = _find_best_checkpoint(checkpoint_dir)
    _save_weights(weight_path, checkpoint_path)

    with open("run/log.txt", "a", encoding="utf-8") as f:
        assert Path("run/log.txt").exists()
        print()
        print(f"INFO: Saved checkpoint '{checkpoint_path}' as weights {weight_path}", file=f)


if __name__ == "__main__":
    main()
