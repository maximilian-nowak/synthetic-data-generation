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


class PyramidBlender(AbstractBlender):
    """
    Pyramid blender class

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
        Blends object with pyramid blending

        Args:
            background: background as a numpy array
            obj: object crop as a numpy array
            mask: object mask as a numpy array
            offset: offset

        Returns:
            Blended object as a numpy array
        """
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Move obj and mask to right position
        p1, p2 = get_bounding_box_with_closing(mask)
        object_width = (p2[0] - p1[0]) / 2
        object_height = (p2[1] - p1[1]) / 2
        shift_x, shift_y = offset[0] - object_width, offset[1] - object_height
        m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        mask_extended = cv2.warpAffine(mask[p1[1]: p2[1], p1[0]: p2[0]], m, (background.shape[1], background.shape[0]))
        obj_extended = cv2.warpAffine(obj[p1[1]: p2[1], p1[0]: p2[0]], m, (background.shape[1], background.shape[0]))

        # prepare mask
        _, mask_extended = cv2.threshold(mask_extended, 254, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask_extended = cv2.erode(mask_extended, kernel, iterations=2)

        mask_extended = (mask_extended // 255).astype("float32")[:, :, np.newaxis]
        mask_extended = np.repeat(mask_extended, 3, axis=2)

        # Replace product background with canvas background
        obj_extended[np.where(mask_extended == 0)] = 0
        obj_extended = np.where(mask_extended == 0, background, obj_extended)

        result = self.laplacian_pyramid_blending_with_mask(obj_extended, background, mask_extended, 3)

        np.clip(result, 0, 255, out=result)

        return result

    @staticmethod
    def laplacian_pyramid_blending_with_mask(foreground, background, mask, num_levels):
        # generate Gaussian pyramid for A,B and mask
        gauss_a = np.float32(foreground.copy())
        gauss_b = np.float32(background.copy())
        gauss_m = np.float32(mask.copy())
        gauss_pyr_a = [gauss_a]
        gauss_pyr_b = [gauss_b]
        gauss_pyr_m = [gauss_m]

        for i in range(num_levels):
            gauss_a = cv2.pyrDown(gauss_a)
            gauss_b = cv2.pyrDown(gauss_b)
            gauss_m = cv2.pyrDown(gauss_m)
            gauss_pyr_a.append(gauss_a)
            gauss_pyr_b.append(gauss_b)
            gauss_pyr_m.append(gauss_m)

        # generate Laplacian Pyramids for A,B and masks
        laplace_pyr_a = [gauss_pyr_a[num_levels - 1]]
        laplace_pyr_b = [gauss_pyr_b[num_levels - 1]]
        gp_mr = [gauss_pyr_m[num_levels - 1]]
        for i in range(num_levels - 1, 0, -1):
            laplace_a = np.subtract(gauss_pyr_a[i - 1], cv2.pyrUp(gauss_pyr_a[i]))
            laplace_b = np.subtract(gauss_pyr_b[i - 1], cv2.pyrUp(gauss_pyr_b[i]))
            laplace_pyr_a.append(laplace_a)
            laplace_pyr_b.append(laplace_b)
            gp_mr.append(gauss_pyr_m[i - 1])  # also reverse the masks

        # Now blend images according to mask in each level
        ls_pyr = []
        for la, lb, gm in zip(laplace_pyr_a, laplace_pyr_b, gp_mr):
            ls = la * gm + lb * (1.0 - gm)
            ls_pyr.append(ls)

        # now reconstruct
        ls_ = ls_pyr[0]
        for i in range(1, num_levels):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, ls_pyr[i])
        ls_[ls_ < 0.0] = 0.0
        ls_[ls_ > 255.0] = 255.0

        return ls_.astype(np.uint8)
