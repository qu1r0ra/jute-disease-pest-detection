from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]
DATA_DIR = ROOT_DIR / "data"
BY_CLASS_DIR = DATA_DIR / "by_class"
ML_SPLIT_DIR = DATA_DIR / "ml_split"

# Image Configuration
IMAGE_SIZE = 640
BATCH_SIZE = 32  # NOTE: Not final.
NUM_WORKERS = 4  # NOTE: Not final.

# Training Configuration
MAX_EPOCHS = 100  # NOTE: Not final.
LEARNING_RATE = 1e-3  # NOTE: Not final.
PATIENCE = 5
SEEDS = [42, 1337, 7, 1234, 99]
DEFAULT_SEED = SEEDS[0]

# Dataset Configuration
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
