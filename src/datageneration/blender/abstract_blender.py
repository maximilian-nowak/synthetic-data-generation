"""
Module with abstract augmenter class

Classes:
    AbstractBlender
"""

from abc import ABCMeta, abstractmethod
from typing import Tuple

import cv2
import numpy as np


class AbstractBlender:
    """
    Abstract blender class

    Methods:
        blend_object
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def blend_object(
        self,
        background: np.ndarray,
        obj: np.ndarray,
        mask: np.ndarray,
        offset: Tuple[int, int],
    ):
        pass


def restore_foreground(img, foreground, background_ori, use_pyramid_blender):
    """
    Restore foreground based on a foreground image with mask
    Args:
        use_pyramid_blender:
        background_ori:
        img:
        foreground:

    Returns: reconstructed foreground in image

    """

    # normalize alpha channels from 0-255 to 0-1
    alpha = foreground[:, :, 3] / 255.0

    if use_pyramid_blender:
        mask = alpha.astype("float32")
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        img = laplacian_pyramid_blending_with_mask(background_ori, img, mask, 2)
    else:
        for c in range(0, 3):
            img[:, :, c] = alpha * background_ori[:, :, c] + img[:, :, c] * (1 - alpha)

    return img


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
    ls_list = []
    for la, lb, gm in zip(laplace_pyr_a, laplace_pyr_b, gp_mr):
        ls = la * gm + lb * (1.0 - gm)
        ls_list.append(ls)

    # now reconstruct
    ls_ = ls_list[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, ls_list[i])
    ls_[ls_ < 0.0] = 0.0
    ls_[ls_ > 255.0] = 255.0

    return ls_.astype(np.uint8)


def fit_brightness_to_scene(obj, scene, mask):
    avg_bkg = np.mean(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))
    avg_obj = np.mean(cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)[mask > 1])
    ratio = min(avg_bkg / avg_obj, 1)
    obj = cv2.convertScaleAbs(obj, alpha=ratio, beta=0)
    return obj
