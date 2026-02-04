import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from jute_disease_pest.utils.constants import DATA_DIR
from jute_disease_pest.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_plant_village(raw_dir: Path, target_dir: Path):
    """Consolidate PlantVillage train/val splits into a single by_class directory."""
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Target directory {target_dir} already exists. Skipping prep.")
        return

    logger.info(f"Preparing PlantVillage dataset in {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["train", "val"]

    for subset in subsets:
        subset_path = raw_dir / subset
        if not subset_path.exists():
            logger.warning(f"Subset folder {subset_path} not found. Skipping.")
            continue
        class_folders = [d for d in subset_path.iterdir() if d.is_dir()]

        for class_folder in tqdm(class_folders, desc=f"Processing {subset}"):
            class_name = class_folder.name
            dest_class_dir = target_dir / class_name
            dest_class_dir.mkdir(exist_ok=True)

            for img_file in class_folder.iterdir():
                # Handle potential duplicate filenames
                if img_file.is_file():
                    dest_file = dest_class_dir / f"{subset}_{img_file.name}"
                    shutil.copy2(img_file, dest_file)

    logger.info("PlantVillage preparation complete!")


def download_plant_village():
    """Download the Plant Village dataset and prepare it."""
    pv_data_dir = DATA_DIR / "plant_village"

    if pv_data_dir.exists() and any(pv_data_dir.iterdir()):
        # Naive check
        if (pv_data_dir / "Apple___Apple_scab").exists():
            logger.info("PlantVillage dataset seems ready. Skipping.")
            return

    logger.info("Downloading PlantVillage dataset...")
    path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
    downloaded_dir = Path(path)
    logger.info(f"Downloaded to {downloaded_dir}")

    raw_source = downloaded_dir / "PlantVillage"
    prepare_plant_village(raw_source, pv_data_dir)


if __name__ == "__main__":
    download_plant_village()
