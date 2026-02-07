from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from jute_disease.utils.constants import IMAGE_SIZE
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


class BaseImgFeatureExtractor(ABC):
    """Abstract base class for image feature extractors."""

    @abstractmethod
    def extract(self, img_pil: Image) -> np.ndarray:
        """Extract features from a PIL image."""
        pass


class ImgRawPixelFeatureExtractor(BaseImgFeatureExtractor):
    """Extracts raw pixel values as features."""

    def __init__(self, img_size: int = IMAGE_SIZE):
        self.img_size = img_size

    def extract(self, img_pil: Image) -> np.ndarray:
        """Resize image and flatten pixel values."""
        img_resized = img_pil.resize(self.img_size)
        img_array = np.array(img_resized)
        return img_array.flatten().astype(np.float32)


def extract_features(
    dataset: ImageFolder,
    extractor: BaseImgFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features using the provided extractor."""
    features_list = []
    labels_list = []

    logger.info(f"Extracting features using {type(extractor).__name__}...")
    for i in tqdm(range(len(dataset)), desc="Feature Extraction"):
        img_pil, label = dataset[i]
        feats = extractor.extract(img_pil)
        features_list.append(feats)
        labels_list.append(label)

    return np.array(features_list), np.array(labels_list)
