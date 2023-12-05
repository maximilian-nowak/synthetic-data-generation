"""
Module with GrabCutCropper

Classes:
    GrabCutCropper
"""
import cv2
import numpy as np

from .abstract_cropper import AbstractCropper


class GrabCutCropper(AbstractCropper):
    """
    GrabCut cropper class

    Methods:
        crop
    """

    def crop(self, image_path: str, output_image: str = "", save_steps: bool = False) -> np.ndarray:
        """
         Crops image with grab cut method and finds boundary of an object on image

        Args:
            image_path: path to image
            output_image: path to output image, if provided it will save there image
            save_steps: states if steps of cropping should be saved

        Returns:
            Array with points of object boundary
        """
        image = cv2.imread(image_path)

        # Calculate object difference to get rough obj position
        bounding_box = self._get_object_position(image)

        # Grabcut
        object_mask = self._grab_cut(np.copy(image), bounding_box)

        object_mask_tuned = self._fine_tune(object_mask)

        contours_final, _ = cv2.findContours(
            object_mask_tuned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        try:
            contours_final = contours_final[0]
        except Exception:
            raise Exception('Cropping error with image', image_path)

        # if save_steps:
        #     image_with_contours = image.copy()
        #     cv2.drawContours(image_with_contours, contours_final, -1, color=255, thickness=10)
        #
        #     contours_image = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        #     cv2.drawContours(contours_image, contours_final, -1, color=255, thickness=10)
        #
        # if output_image:
        #     contours_image = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        #     cv2.drawContours(contours_image, contours_final, -1, color=255, thickness=10)
        #     cv2.imwrite(output_image, contours_image)

        return contours_final

    @staticmethod
    def _fine_tune(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask_fine_tuned = mask.copy()
        mask_fine_tuned = cv2.erode(mask_fine_tuned, kernel, iterations=10)
        mask_fine_tuned = cv2.dilate(mask_fine_tuned, kernel, iterations=3)
        return mask_fine_tuned

    @staticmethod
    def _grab_cut(image, bounding_box) -> np.ndarray:

        height, width = image.shape[:2]
        mask_result = np.zeros((height, width), dtype=np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask_result, bounding_box, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

        mask_result = np.where(
            (mask_result == cv2.GC_PR_BGD) | (mask_result == cv2.GC_BGD), 0, 1
        ).astype("uint8")

        return mask_result

    @staticmethod
    def _get_object_position(image):

        # Todo: Add gaussian blurring 7x7 to image to improve thresholding
        # Todo: Maybe use cv2.ADAPTIVE_THRESH_MEAN_C
        thresh_mask = cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            maxValue=1,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=9,
            C=4,
        )

        thresh_mask = cv2.erode(thresh_mask, np.ones((3, 3), np.uint8), iterations=1)
        thresh_mask = cv2.dilate(thresh_mask, np.ones((25, 25), np.uint8), iterations=10)
        thresh_mask = cv2.erode(thresh_mask, np.ones((5, 5), np.uint8), iterations=10)

        contours, _ = cv2.findContours(
            thresh_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_box = cv2.boundingRect(contours[0])

        return bounding_box
