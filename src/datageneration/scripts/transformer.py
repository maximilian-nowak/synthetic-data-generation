"""
Module that wraps classes for augmentations

Functions:
    apply_augmentations
"""
from typing import List, Tuple

import cv2
import numpy as np


def apply_3d_transformation(
    image: np.ndarray, mask: np.ndarray, background_config: List[str], offset,
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Function that applies 3d transformation to given image according to given cord and
    background_config.

    Args:
        image: image that will be augment
        mask: masking image for augmentation
        background_config: config of background image
        offset: (x,y)

    Returns:
        Tuple (image, mask, offset) where image is augmented image and mask is mask used for augment.

    """
    if background_config is None or background_config == "":
        return image, mask, offset

    if "big" in background_config:
        image, mask, offset = transform(image, mask, offset)

    return image, mask, offset


def transform(image, mask, offset):
    height, width, _ = image.shape

    position_y_ratio = offset[1]/height

    if position_y_ratio > 0.5:

        # tilt y
        tilt = int(0.04 * position_y_ratio * width)

        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst_points = np.float32([[0, 0], [width, 0], [width - tilt, height], [tilt, height]])
        # Compute the perspective transformation matrix
        m = cv2.getPerspectiveTransform(src_points, dst_points)
        # Warp image and mask according to transformation matrix
        tilted_image = cv2.warpPerspective(image, m, (width, height), borderMode=cv2.BORDER_CONSTANT)
        tilted_mask = cv2.warpPerspective(mask, m, (width, height))

        # resize image for a better fit
        resized_image = cv2.resize(tilted_image, (width, int(height*0.7)))
        resized_mask = cv2.resize(tilted_mask, (width, int(height*0.7)))

        # correct offset to accommodate smaller image
        corrected_offset = (offset[0], offset[1]+int(height*0.1))

    else:
        return image, mask, offset

    return resized_image, resized_mask, corrected_offset
