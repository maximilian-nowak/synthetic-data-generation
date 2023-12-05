"""
Module with random rotation augmentation functionalities

Classes:
    RandomRotationAugmenter
"""

from random import randint
from typing import Tuple

import numpy as np
from PIL import Image

from .abstract_augmenter import AbstractAugmenter


class RandomRotationWithOutliersAugmenter(AbstractAugmenter):
    """
    Class that aggregates methods for image random rotation.

    Methods:
         augment
         get_random_angle
    """

    def augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augments image.

        Args:
            image: image in form of numpy array
            mask: image mask in form of numpy array

        Returns:
            Tuple of 2 numpy arrays: randomly rotated image and mask.
        """
        angle = self.get_random_angle()
        rotated_image = self.rotate(image, angle, "white")
        rotated_mask = self.rotate(mask, angle, "black")
        return rotated_image, rotated_mask

    @staticmethod
    def get_random_angle() -> int:
        choice = randint(0, 100)
        if choice > 95:
            return 90
        elif choice > 90:
            return -90
        elif choice > 70:
            return randint(-50, 50)
        else:
            return randint(-2, 2)

    @staticmethod
    def rotate(image, angle, fillcolor):
        img = Image.fromarray(image)
        out = img.rotate(angle, expand=True, fillcolor=fillcolor)
        return np.array(out)
