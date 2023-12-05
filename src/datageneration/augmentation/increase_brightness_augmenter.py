"""
Module with brightness change functionalities

Classes:
    IncreaseBrightnessAugmenter
"""

from typing import Tuple

import cv2
import numpy as np

from .abstract_augmenter import AbstractAugmenter


class IncreaseBrightnessAugmenter(AbstractAugmenter):
    """
    Class that aggregates methods for image brightness increase

    Methods:
        augment
        get_hsv_channels
        hsv_to_bgr
    """

    def augment(
        self, image: np.ndarray, mask: np.ndarray, value: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augments image.

        Args:
            image: image in form of numpy array
            mask: image mask in form of numpy array
            value: addition to each pixel to make image brighter

        Returns:
            Tuple of 2 numpy arrays: brighter image and mask used.
        """
        # reduce contrast by 50%
        image = np.uint8(0.5 * image)
        h, s, v = self.get_hsv_channels(image)
        # reduce saturation by 20%
        s = np.uint8(0.8 * s)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        brighter_image = self.hsv_to_bgr(h, s, v)

        return brighter_image, mask

    @staticmethod
    def get_hsv_channels(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Changes image palette from BGR to HSV and splits image channels to separate variables

        Args:
            image: image in form of numpy array and with BGR palette

        Returns:
            Tuple of 3 numpy array related to 3 h, s and v channels.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        return h, s, v

    @staticmethod
    def hsv_to_bgr(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Merges h, s and v channels into one image and convert it to BGR

        Args:
            h: hue channel numpy array
            s: saturation channel numpy array
            v: value channel numpy array

        Returns:
            Image in BGR palette
        """
        image_hsv = cv2.merge((h, s, v))
        image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        return image_bgr
