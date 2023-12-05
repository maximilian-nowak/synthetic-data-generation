"""
Module with abstract cropper

Classes:
    AbstractCropper
"""
from abc import abstractmethod


class AbstractCropper:
    """
    Abstract cropper class

    Methods:
        crop
        compute_alpha_shape
        alpha_shape
    """

    @abstractmethod
    def crop(self, image_path: str, output_image: str = "", save_steps: bool = False):
        pass
