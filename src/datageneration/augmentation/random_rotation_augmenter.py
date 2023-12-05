"""
Module with random rotation augmentation functionalities

Classes:
    RandomRotationAugmenter
"""

from random import randint
from typing import Tuple

import cv2
import numpy as np

from .abstract_augmenter import AbstractAugmenter


class RandomRotationAugmenter(AbstractAugmenter):
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
        rotated_image = self.rotate(image, angle, (255, 255, 255))
        rotated_mask = self.rotate(mask, angle, (0, 0, 0))

        return rotated_image, rotated_mask

    @staticmethod
    def get_random_angle() -> int:
        return randint(0, 360)

    @staticmethod
    def rotate(image, angle, border_value, center=None, scale=1.0):
        # grab the dimensions of the image
        (h, w) = image.shape[:2]

        # if the center is None, initialize it as the image center
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=border_value)

        # return the rotated image
        return rotated
