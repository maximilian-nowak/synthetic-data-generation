"""
Module with none cropper

Classes:
    NoneCropper
"""
from math import ceil
from typing import Tuple

import cv2
import numpy as np

from .abstract_blender import AbstractBlender


class AlphaBlender(AbstractBlender):
    """
    Alpha blender class

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
        Blends object with alpha

        Args:
            background: background as a numpy array
            obj: object crop as a numpy array
            mask: object mask as a numpy array
            offset: offset

        Returns:
            Blended object as a numpy array
        """
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        obj = cv2.bitwise_and(mask, obj)

        # Trim mask and object
        x0, x1, y0, y1 = self.get_trimmed_coords(mask)
        trimmed_mask = mask[x0:x1, y0:y1]
        trimmed_object = obj[x0:x1, y0:y1]

        background_width = background.shape[1]
        background_height = background.shape[0]
        trimmed_width = float(trimmed_mask.shape[1])
        trimmed_height = float(trimmed_mask.shape[0])

        left = int(ceil(offset[0] - trimmed_width / 2))
        right = int(background_width - offset[0] - trimmed_width / 2)
        top = int(ceil(offset[1] - trimmed_height / 2))
        bottom = int(background_height - offset[1] - trimmed_height / 2)

        bottom = max(bottom, 0)
        left = max(left, 0)
        top = max(top, 0)
        right = max(right, 0)

        # Move object by offset
        black = [0, 0, 0]
        obj = cv2.copyMakeBorder(
            trimmed_object, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black
        )
        mask = cv2.copyMakeBorder(
            trimmed_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black
        )

        # Dilate mask
        mask = self.prepare_mask(mask)

        # Normalize the alpha mask to keep intensity between 0 and 1
        mask = mask.astype(float) / 255
        obj = obj.astype(float)
        background = background.astype(float)

        # Multiply the foreground with the alpha matte
        obj = cv2.multiply(mask, obj)

        # Multiply the background with ( 1 - alpha )
        if mask.shape != background.shape:
            mask = cv2.resize(mask, (background_width, background_height))
        background = cv2.multiply(1.0 - mask, background)

        # Add the masked foreground and background.
        if obj.shape != background.shape:
            obj = cv2.resize(obj, (background_width, background_height))
        out_image = cv2.add(obj, background)

        return out_image.astype(np.uint8)

    @staticmethod
    def prepare_mask(mask: np.ndarray) -> np.ndarray:
        """
        Dilates image with OpenCV methods

        Args:
            mask: image as numpy array

        Returns:
            Dilated image as numpy array
        """
        # Prepare mask if binary
        if len(np.unique(mask)) > 2:
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return mask

    @staticmethod
    def get_trimmed_coords(image: np.ndarray, tolerance: int = 0) -> Tuple[int, int, int, int]:
        """
        Trim coordinates of non-black pixels

        Args:
            image: image as a numpy array
            tolerance: threshold for trimming non-black pixels

        Returns:
            Tuple (x0, x1, y0, y1) with coordinates of non-black pixels bounding box
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > tolerance

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

        return x0, x1, y0, y1
