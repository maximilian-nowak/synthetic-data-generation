"""
Module that wraps classes for augmentations

Functions:
    apply_augmentations
"""
import random
from typing import List, Tuple

import numpy as np

from .increase_brightness_augmenter import IncreaseBrightnessAugmenter
from .random_blur_augmenter import RandomBlurAugmenter
from .random_rotation_augmenter import RandomRotationAugmenter
from .random_rotation_limited_augmenter import RandomRotationLimitedAugmenter
from .random_rotation_withoutliers_augmenter import RandomRotationWithOutliersAugmenter


def apply_augmentations(image: np.ndarray, mask: np.ndarray, augmentations_list: List[str]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that applies augmentations to given image according to given mask and selected
    augmentations list.

    Args:
        image: image that will be augment
        mask: masking image for augmentation
        augmentations_list: list of augmentations that will be used in process

    Returns:
        Tuple (image, mask) where image is augmented image and mask is mask used for augment.

    Raises:
        AttributeError: raises when wrong name of augmentation is given.
    """
    for augmentation in augmentations_list:
        if random.randrange(0, 200) <= 100/len(augmentations_list):
            augmenter = augmentations[augmentation]
            image, mask = augmenter.augment(image, mask)

    return image, mask


augmentations = {
    "increase_brightness": IncreaseBrightnessAugmenter(),
    "random_blur": RandomBlurAugmenter(),
    "random_rotation": RandomRotationAugmenter(),
    "random_rotation_limited": RandomRotationLimitedAugmenter(),
    "random_rotation_with-outliers": RandomRotationWithOutliersAugmenter(),
}
