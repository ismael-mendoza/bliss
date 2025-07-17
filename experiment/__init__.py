"""Replace the following paths with where you want to save figures, datasets, etc."""

from pathlib import Path

SEED = 52

DATASETS_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/datasets/")
CACHE_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/cache/")
FIGURE_DIR = Path("/home/imendoza/bliss/experiment/figures/")
MODELS_DIR = Path("/home/imendoza/bliss/experiment/models/")

# metadata for models saved for pytorch lightning
TORCH_DIR = Path("/nfs/turbo/lsa-regier/scratch/ismael/out/")
