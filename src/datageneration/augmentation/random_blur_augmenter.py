"""
Module with random blur augmentation functionalities

Classes:
    RandomBlurAugmenter
"""

from random import choice, randrange
from typing import Tuple

import cv2
import numpy as np

from .abstract_augmenter import AbstractAugmenter


class RandomBlurAugmenter(AbstractAugmenter):
    """
    Class that aggregates methods for image random blur augmentation. It chooses kernel for blur
    randomly.

    Methods:
         augment
         decision
         blur
    """

    kernel = choice([(3, 3), (5, 5), (7, 7)])
    sigma_x = 10

    def augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augments image.

        Args:
            image: image in form of numpy array
            mask: image mask in form of numpy array

        Returns:
            Tuple of 2 numpy arrays: blurred (or not) image and mask used.
        """
        if self.decision(10):
            image = self.blur(image, self.kernel)

        return image, mask

    @staticmethod
    def decision(percent: int = 50) -> bool:
        """
        Random decision generator

        Args:
            percent: threshold of decision to be positive

        Returns:
            Returns if decision was True or False
        """
        return randrange(100) < percent

    def blur(self, image: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
        """
        Does blur opencv operation on image with proper parameters.

        Args:
            image: image in form of numpy array
            kernel: kernel for gaussian blur

        Returns:
            Blurred image.
        """
        blurred = cv2.GaussianBlur(image, kernel, self.sigma_x)

        return blurred
