"""
Module with image processing functions for data generation module

Functions:
    resize
    resize_image_aspect_ratio
    blur
    add_gaussian_noise
"""
from typing import Optional

import cv2
import numpy as np


def resize(image: np.ndarray, size: float) -> np.ndarray:
    """
    Resizes image by percentage ratio given.

    Args:
        image: image in form of numpy array
        size: resize ratio

    Returns:
        Resized image.
    """
    width = image.shape[1]
    height = image.shape[0]

    new_width = int(width * size / 100)
    new_height = int(height * size / 100)

    # image size can not be odd
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)

    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image_resized


def resize_image_aspect_ratio(
    image: np.ndarray, new_width: Optional[int] = None, new_height: Optional[int] = None
) -> np.ndarray:
    """
    Resizes image by calculate aspect ratio of new width or new height given.

    Args:
        image: image in form of numpy array
        new_width: new width that image will have
        new_height: new height that image will have

    Returns:
        Resized image.
    """
    height, width = image.shape[:2]

    r = 1.0
    if new_width is not None and new_height is None:
        r = new_width / width
    elif new_width is None and new_height is not None:
        r = new_height / height

    new_height = int(height * r)
    new_width = int(width * r)
    new_image = cv2.resize(image, (new_width, new_height))

    return new_image


def blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """
    Creates gaussian noise and adds it to image

    Args:
        image: image in form of numpy array

    Returns:
        Image with addition of random gaussian noise
    """
    row, col, ch = image.shape

    mean = 0
    var = 0.1
    sigma = var ** 0.5

    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape((row, col, ch))

    return image + gauss
