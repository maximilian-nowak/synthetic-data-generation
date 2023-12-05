"""
Module with files functionalities
"""

import glob
import os.path
import random
from typing import Dict, List

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}


def get_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, "*"))


def get_background_images(directory: str) -> List[str]:
    images = []
    files = get_files(directory)
    for file in files:
        if "_placement-area" not in str(file) and "_front" not in str(file):
            images.append(file)

    return images


def get_object_images(directory: str) -> Dict[str, List[str]]:
    training_images = {}

    training_directories = glob.glob(directory + "/*")
    for training_directory in training_directories:
        files_paths = [file for file in get_files(training_directory)]
        class_name = os.path.basename(os.path.normpath(training_directory))
        for file_path in files_paths:
            if class_name in training_images:
                training_images[class_name].append(file_path)
            else:
                training_images[class_name] = [file_path]

    return training_images


def get_extension(path_to_file: str) -> str:
    return str(path_to_file.rpartition(".")[-1])


def get_angled_image_from_folder(folder_path: str, background_config: str):

    images = get_files(folder_path)

    if background_config is None or background_config == [""]:
        return random.choice(images)

    for i in background_config:
        if "angle" in i:
            angle = int(i.split(':')[-1])
            possible_angles = [angle, angle + 9, angle + 18, angle + 27]

            matching_imgs = []
            for img in images:
                img_file_name = os.path.basename(img)
                for possible_angle in possible_angles:
                    if 'H' + str(possible_angle * 10) + 'V30' in img_file_name:
                        matching_imgs.append(img)

            if len(matching_imgs) > 0:
                return random.choice(matching_imgs)
            else:
                return random.choice(images)


def get_backgrounds(background_path: str, use_placement_area: bool, reset_foreground: bool):
    """
    Load Backgrounds

    Args:
        use_placement_area
        background_path
        reset_foreground

    Returns:
        background,
        allowed_areas_mask,
        foreground
    """

    background = np.array(cv2.imread(background_path))

    # Get Background mask for allowed placing areas
    if use_placement_area:

        if ".jpg" in background_path:
            allowed_areas_path = background_path.replace(".jpg", "_placement-area.png")
        else:
            allowed_areas_path = background_path.replace(".png", "_placement-area.png")

        if not os.path.isfile(allowed_areas_path):
            raise ValueError("There is no allowed_area defined")

        allowed_areas_mask = cv2.imread(allowed_areas_path, cv2.IMREAD_UNCHANGED)
        if allowed_areas_mask.shape[2] < 4:
            raise Exception(
                background_path + " --> placement-area.png does not have a alpha channel"
            )
        _, allowed_areas_mask = cv2.threshold(
            allowed_areas_mask[:, :, 3], 0, 1, cv2.THRESH_BINARY_INV
        )
    else:
        allowed_areas_mask = None

    # Get Foreground if available
    if reset_foreground:
        if ".jpg" in background_path:
            foreground_path = background_path.replace(".jpg", "_front.png")
        else:
            foreground_path = background_path.replace(".png", "_front.png")

        if not os.path.exists(foreground_path):
            raise ValueError("There is no foreground defined")

        foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
        if foreground.shape[2] != 4:
            raise ValueError(
                background_path + " -->foreground image does not have a alpha layer"
            )
    else:
        foreground = None

    config_path = background_path.replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = f.read().splitlines()
    else:
        config = [""]

    # Correct Background format to be odd
    height, width, _ = background.shape
    # Calculate the new width and height
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    # Calculate the crop box coordinates
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    # Crop the image
    background = background[top:bottom, left:right, :]

    if foreground is not None:
        foreground = foreground[top:bottom, left:right, :]

    return background, allowed_areas_mask, foreground, config
