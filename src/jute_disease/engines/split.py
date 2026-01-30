import random
import shutil

from tqdm import tqdm

from jute_disease.utils.constants import (
    BY_CLASS_DIR,
    DEFAULT_SEED,
    ML_SPLIT_DIR,
    SPLITS,
)
from jute_disease.utils.seed import seed_everything


def split_dataset():
    seed_everything(DEFAULT_SEED)

    if not BY_CLASS_DIR.exists():
        print(f"Error: Source directory {BY_CLASS_DIR} does not exist.")
        return

    class_folders = [f for f in BY_CLASS_DIR.iterdir() if f.is_dir()]

    if not class_folders:
        print(f"Error: No class folders found in {BY_CLASS_DIR}")
        return

    print(f"Found {len(class_folders)} classes. Starting split...")

    for class_folder in class_folders:
        class_name = class_folder.name

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            images.extend(list(class_folder.glob(ext)))

        if not images:
            print(f"Warning: No images found for class '{class_name}'. Skipping.")
            continue

        random.shuffle(images)

        num_images = len(images)
        train_idx = int(num_images * SPLITS["train"])
        val_idx = train_idx + int(num_images * SPLITS["val"])

        assignments = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:],
        }

        print(f"Processing '{class_name}': {num_images} images")

        for split, split_images in assignments.items():
            split_path = ML_SPLIT_DIR / split / class_name
            split_path.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_images, desc=f"  -> {split}", leave=False):
                shutil.copy2(img_path, split_path / img_path.name)

    print("\nData splitting complete!")


if __name__ == "__main__":
    split_dataset()
