"""
Module with seamless blender

Classes:
    SeamlessBlender
"""
from typing import Tuple

import cv2
import numpy as np

from .abstract_blender import AbstractBlender


class SeamlessBlender(AbstractBlender):
    """
    Seamless blender class

    Methods:
        blend_object
    """

    def blend_object(
        self,
        background: np.ndarray,
        obj: np.ndarray,
        mask: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """
        Blends object with seamlessClone OpenCV method

        Args:
            background: background as a numpy array
            obj: object crop as a numpy array
            mask: object mask as a numpy array
            offset: offset

        Returns:
            Blended object as a numpy array
        """

        # prepare mask
        _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

        # Add Border around Images to bake Blending more stable
        border = 250
        obj = cv2.copyMakeBorder(
            obj,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
        mask = cv2.copyMakeBorder(
            mask, border, border, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        background_ext = cv2.copyMakeBorder(
            background,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=[180, 180, 180],
        )

        # replace obj background color with background of background
        offset_y_1 = max(int(offset[1] - obj.shape[0] / 2)+border, 0)
        offset_y_2 = int(offset[1] + obj.shape[0] / 2)+border
        offset_x_1 = max(int(offset[0] - obj.shape[1] / 2)+border, 0)
        offset_x_2 = int(offset[0] + obj.shape[1] / 2)+border

        # Check if bounding box would reach over background
        background_ext_box = background_ext[offset_y_1:offset_y_2, offset_x_1:offset_x_2]
        if obj.shape[0] > background_ext_box.shape[0] or obj.shape[1] > background_ext_box.shape[1]:
            obj = obj[0:background_ext_box.shape[0], 0:background_ext_box.shape[1]]
            mask = mask[0:background_ext_box.shape[0], 0:background_ext_box.shape[1]]
        mask_extended_2 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        obj = np.where(mask_extended_2 == (0, 0, 0), background_ext[offset_y_1:offset_y_2, offset_x_1:offset_x_2], obj)

        # Dilate mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)

        offset = (offset[0] + border, offset[1] + border)
        output = cv2.seamlessClone(obj, background_ext, mask, offset, cv2.NORMAL_CLONE)
        output = output[border:-border, border:-border]  # Remove Border from Background image

        return output

