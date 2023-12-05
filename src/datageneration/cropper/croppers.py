"""
Module with all croppers and tools

Functions:
    extract_object
    remove_background
    make_mask
    get_bounding_box
"""
from typing import Tuple

import cv2
import numpy as np

from src.datageneration.cropper.grab_cut_cropper import GrabCutCropper

def extract_object(image: np.ndarray, polygon: np.ndarray, boundary: float = 0.1) -> np.ndarray:
    """
    Gets rectangle with extreme points from given polygon and crops rectangle from original image.

    Args:
        image: image in form of numpy array
        polygon: numpy array with all polygon points
        boundary: boundary in percent that will be added to object rectangle to exceed its
                  coordinates

    Returns:
        Cropped object image
    """
    x, y, w, h = cv2.boundingRect(np.array(polygon))
    relative_boundary = min(boundary * w, boundary * h)

    x -= int(relative_boundary)
    y -= int(relative_boundary)
    w += int(2 * relative_boundary)
    h += int(2 * relative_boundary)

    cropped_object = image[y: y + h, x: x + w]

    return cropped_object


def remove_background(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Removes background apart from what is inside of polygon

    Args:
        image: image in form of numpy array
        polygon: numpy array with all polygon points

    Returns:
        Image with removed background apart from object
    """
    height = image.shape[0]
    width = image.shape[1]

    blank_image = np.ones((height, width, 3), np.uint8) * 255
    markers = cv2.fillPoly(blank_image, pts=[polygon], color=(0, 0, 0))

    return cv2.bitwise_or(image, markers)


def extract_mask(image_path: str, cropper) -> np.ndarray:
    """
    Extract mask with selected cropper

    Args:
        image_path: path to image to extract mask
        cropper: cropper object

    Returns:
        Mask
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        mask = image[:, :, 3]

        # Check if mask is correct
        if mask.mean().mean() < 0.1:
            raise Exception('Mask is empty with', image_path)

    else:
        pts = cropper.crop(image_path)
        mask = make_mask(image, pts)

    return mask


def make_mask(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Makes mask of object on image according to polygon points

    Args:
        image: image in form of numpy array
        pts: numpy array with all polygon points

    Returns:
        Mask for object extraction in form of numpy array
    """
    width = image.shape[1]
    height = image.shape[0]

    mask = np.zeros((height, width, 1), np.uint8)

    pts = np.array(pts, np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    return mask


def get_bounding_box(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Creates bounding box from mask of object

    Args:
        mask: mask of object on image

    Returns:
        Tuple of points, where each point is tuple of x and y coordinates.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(cnts[0])
    assert len(cnts) == 1
    return (x, y), (x + w, y + h)


def get_bounding_box_with_closing(
    mask: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Creates bounding box from mask of object

    Args:
        mask: mask of object on image

    Returns:
        Tuple of points, where each point is tuple of x and y coordinates.
    """

    kernel = np.ones((15, 15), np.uint8)
    mask_closing = cv2.dilate(mask, kernel, iterations=4)
    mask_closing = cv2.erode(mask_closing, kernel, iterations=4)

    return get_bounding_box(mask_closing)


def get_bounding_box_list(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Creates list of bounding boxes of objects from mask

    Args:
        mask: mask of object on image

    Returns:
        List of Tuple of points, where each point is tuple of x and y coordinates.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(((x, y), (x + w, y + h)))

    return boxes


def limit_size_of_obj(object_image: np.array, mask: np.array):
    """
     Cut object image to a larger bounding box around image

    Args:
        object_image:
        mask:

    Returns: cut object_image, cut object_image

    """
    kernel = np.ones((40, 40), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        raise Exception("Too many contours in limit_size_of_obj")

    x_min, y_min, w, h = cv2.boundingRect(contours[0])
    x_max = x_min + w
    y_max = y_min + h
    x_min -= int(0.1 * (x_max - x_min))
    x_max += int(0.1 * (x_max - x_min))
    y_min -= int(0.1 * (y_max - y_min))
    y_max += int(0.1 * (y_max - y_min))

    x_min = max(0, x_min)
    y_min = max(0, y_min)

    height, width = object_image.shape[:2]
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    object_image_cropped = object_image[y_min:y_max, x_min:x_max]
    mask_cropped = mask[y_min:y_max, x_min:x_max]
    return object_image_cropped, mask_cropped


available_croppers = {
    "grabcut": GrabCutCropper(),
}
