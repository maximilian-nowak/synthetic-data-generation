"""
Module with none cropper

Classes:
    NoneCropper
"""
from typing import Tuple

import cv2
import numpy as np

from ..cropper.croppers import get_bounding_box_with_closing
from .abstract_blender import AbstractBlender


class NoBlender(AbstractBlender):
    """
    No blender class

    Methods:
        blend_object
        get_trimmed_coords
        dilate
    """

    def blend_object(
        self,
        background: np.ndarray,
        obj: np.ndarray,
        mask: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """
        Blends object with alpha

        Args:
            background: background as a numpy array
            obj: object crop as a numpy array
            mask: object mask as a numpy array
            offset: offset

        Returns:
            Blended object as a numpy array
        """

        # Move obj and mask to right position
        p1, p2 = get_bounding_box_with_closing(mask)
        object_width = (p2[0] - p1[0]) / 2
        object_height = (p2[1] - p1[1]) / 2
        shift_x = offset[0] - object_width
        shift_y = offset[1] - object_height
        m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        mask_extended = cv2.warpAffine(
            mask[p1[1]: p2[1], p1[0]: p2[0]],
            m,
            (background.shape[1], background.shape[0]),
        )
        obj_extended = cv2.warpAffine(
            obj[p1[1]: p2[1], p1[0]: p2[0]],
            m,
            (background.shape[1], background.shape[0]),
        )

        # Insert object into background
        mask_extended[mask_extended > 0] = 255
        mask_extended = np.repeat(mask_extended[:, :, np.newaxis], 3, axis=2)
        out_image = np.where(mask_extended == (255, 255, 255), obj_extended, background)

        return out_image
