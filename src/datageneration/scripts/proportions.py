"""
Module consists of classes for proportion calculation

Classes:
    AbstractProportionsComputator
    RelativeProportionsComputator
    UniformProportionsComputator
"""

from abc import ABCMeta, abstractmethod
from math import sqrt
from typing import Tuple

import cv2
import numpy as np

from src.datageneration.cropper.croppers import get_bounding_box_with_closing


class AbstractProportionsComputator:
    """
    Abstract proportion computation class

    Methods:
        compute_proportion
        get_object_size
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def compute_proportion(
        self, mask: np.ndarray, background: np.ndarray, desirable_proportion: int
    ):
        pass

    def get_object_size(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Get object image size

        Args:
            mask: object mask as numpy array

        Returns:
            Tuple with object width and height
        """
        p1, p2 = get_bounding_box_with_closing(mask)
        object_width = p2[0] - p1[0]
        object_height = p2[1] - p1[1]

        return object_width, object_height


class RelativeProportionsComputator(AbstractProportionsComputator):
    """
    Class for computation of relative proportions

    Methods:
        compute_proportion
    """

    def compute_proportion(
        self, mask: np.ndarray, background: np.ndarray, desirable_proportion: int
    ) -> int:
        """
        Computes relative proportion

        Args:
            mask: object mask as numpy array
            background: background image as numpy array
            desirable_proportion: proportion that should be achieved

        Returns:
            Relative proportion
        """
        object_width, object_height = self.get_object_size(mask)

        background_width = background.shape[1]
        background_height = background.shape[0]

        current_proportions_width = float(object_width) / background_width * 100
        current_proportions_height = float(object_height) / background_height * 100

        desirable_object_width = float(
            (object_width * desirable_proportion) / current_proportions_width
        )
        desirable_object_height = float(
            (object_height * desirable_proportion) / current_proportions_height
        )

        relative_proportion_width = int(desirable_object_width * 100 / object_width)
        relative_proportion_height = int(desirable_object_height * 100 / object_height)

        relative_proportion = min(relative_proportion_width, relative_proportion_height)

        return relative_proportion


class UniformProportionsComputator(AbstractProportionsComputator):
    """
    Class for computation of uniform proportions

    Methods:
        compute_proportion
    """

    def compute_proportion(
        self, mask: np.ndarray, background: np.ndarray, desirable_proportion: int
    ) -> int:
        """
        Computes uniform proportion

        Args:
            mask: object mask as numpy array
            background: background image as numpy array
            desirable_proportion: proportion that should be achieved

        Returns:
            Uniform proportion
        """
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=5)
        object_width, object_height = self.get_object_size(mask_dilated)
        object_area = object_width * object_height
        mask_area = mask.shape[1] * mask.shape[0]
        background_area = background.shape[1] * background.shape[0]

        desirable_area = object_area * background_area / mask_area
        relative_size = float(sqrt(desirable_area)) / sqrt(object_area) * 100

        uniform_proportion = int(relative_size * desirable_proportion / 100)

        return uniform_proportion


available_proportions_computator = {
    "relative": RelativeProportionsComputator(),
    "uniform": UniformProportionsComputator(),
}
