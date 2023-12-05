"""
Module with classes used for placement calculation

Classes:
    AbstractPlacer
    PlacementMapPlacer
    IoUCheckingPlacer
"""
import random
from abc import ABCMeta, abstractmethod
from random import randint
from typing import List, Tuple

import cv2
import numpy as np

from src.datageneration.cropper.croppers import get_bounding_box_list, get_bounding_box_with_closing

BoxType = List[Tuple[int, int]]


class AbstractPlacer:
    """
    Abstract placer class
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.placement_map = None

    @abstractmethod
    def compute_offsets(
            self,
            mask: np.ndarray,
            background: np.ndarray,
            background_mask: np.ndarray,
            use_placement_area: bool,
    ) -> Tuple[int, int]:
        pass

    @abstractmethod
    def compute_offsets_with_group_placement(
        self,
        mask: np.ndarray,
        background: np.ndarray,
        background_mask: np.ndarray,
        class_idx: int,
    ):
        pass


class IoUCheckingPlacer(AbstractPlacer):
    """
    Placer class with IoU checking condition

    Methods:
        compute_offsets
        get_iou
        check_location
        compute_offsets_with_group_placement
    """

    def __init__(self, max_iou: float = 0.0):
        super().__init__()

        self.locations = []
        self.max_iou = max_iou
        self.max_iters_for_placement = 50

    def compute_offsets(
        self,
        mask: np.ndarray,
        background: np.ndarray,
        background_mask: np.ndarray,
        use_placement_area: bool,
    ) -> Tuple[int, int]:

        background_height, background_width = background.shape[:2]

        # Placement area from background_mask
        if use_placement_area:
            area_boxes = get_bounding_box_list(background_mask)
            allowed_area_p1, allowed_area_p2 = random.choice(
                area_boxes
            )  # Select random area in which obj should be placed
            allowed_area_width = allowed_area_p2[0] - allowed_area_p1[0]
            allowed_area_height = allowed_area_p2[1] - allowed_area_p1[1]

        p1, p2 = get_bounding_box_with_closing(mask)
        mask_w = p2[0] - p1[0]
        mask_h = p2[1] - p1[1]
        mask_w_half = int(mask_w / 2)
        mask_h_half = int(mask_h / 2)

        height_mask_offset = 100
        width_mask_offset = 100

        for _ in range(self.max_iters_for_placement):

            if use_placement_area:
                # Generate random positions inside allowed area
                if mask_w_half <= allowed_area_width - mask_w:
                    offset_x = int(
                        randint(mask_w_half, allowed_area_width - mask_w)
                        + allowed_area_p1[0]
                        - height_mask_offset
                        + mask_w_half
                    )
                else:
                    offset_x = int(
                        allowed_area_width / 2
                        + allowed_area_p1[0]
                        - height_mask_offset
                        + mask_w_half
                    )
                offset_y = int(
                    randint(0, allowed_area_height)
                    + allowed_area_p1[1]
                    - width_mask_offset
                    - mask_h_half
                )
            else:
                offset_x = randint(int(background_width*0.25), int(background_width*0.75))
                offset_y = randint(int(background_height*0.25), int(background_height*0.75))
                #offset_x = randint(250, background_width - 250)
                #offset_y = randint(350, background_height - 350)

            new_p1 = (offset_y - mask_h_half, offset_x - mask_w_half)
            new_p2 = (offset_y + mask_h_half, offset_x + mask_w_half)
            new_location = [new_p1, new_p2]

            if self.check_inside_image(
                offset_x, offset_y, background_width, background_height
            ) and self.check_location(new_location, self.max_iou):
                self.locations.append(new_location)
                return offset_y, offset_x

        return -1, -1

    @staticmethod
    def get_iou(box_a: BoxType, box_b: BoxType) -> float:
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0][0], box_b[0][0])
        y_a = max(box_a[0][1], box_b[0][1])
        x_b = min(box_a[1][0], box_b[1][0])
        y_b = min(box_a[1][1], box_b[1][1])

        # compute the area of intersection rectangle
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[1][0] - box_a[0][0] + 1) * (box_a[1][1] - box_a[0][1] + 1)
        box_b_area = (box_b[1][0] - box_b[0][0] + 1) * (box_b[1][1] - box_b[0][1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        # return the intersection over union value
        return iou

    @staticmethod
    def check_inside_image(offset_x, offset_y, background_width, background_height):
        return 0 < offset_x < background_width and 0 < offset_y < background_height

    def check_location(self, new_location: BoxType, max_iou: float) -> bool:

        for loc in self.locations:
            iou = self.get_iou(loc, new_location)
            if iou > max_iou:
                return False

        return True

    def compute_offsets_with_group_placement(
        self,
        mask: np.ndarray,
        background: np.ndarray,
        background_mask: np.ndarray,
        class_idx: int,
    ) -> Tuple[int, int]:

        # ToDo: Placement logic of allowed area missing

        background_width = background.shape[1]
        background_height = background.shape[0]

        if self.placement_map is None:
            self.placement_map = np.zeros((background_height, background_width, 1), np.uint8)

        p1, p2 = get_bounding_box_with_closing(mask)
        mask_w = p2[0] - p1[0]
        mask_h = p2[1] - p1[1]

        offset_x = 0
        offset_y = 0

        nonzeros_x = np.where(self.placement_map == class_idx)[0]
        nonzeros_y = np.where(self.placement_map == class_idx)[1]

        filtered_nonzeros_x = np.array(
            filter(
                lambda x: int(mask_w / 2) < x < (background_width - int(mask_w / 2)),
                nonzeros_x,
            )
        )
        filtered_nonzeros_y = np.array(
            filter(
                lambda y: int(mask_h / 2) < y < (background_height - int(mask_h / 2)),
                nonzeros_y,
            )
        )

        for _ in range(self.max_iters_for_placement):
            if len(filtered_nonzeros_x) and len(filtered_nonzeros_y):
                random_x = randint(0, len(filtered_nonzeros_x) - 1)
                random_y = randint(0, len(filtered_nonzeros_y) - 1)

                new_offset_x = filtered_nonzeros_x[random_x]
                new_offset_y = filtered_nonzeros_y[random_y]
            else:
                new_offset_x = randint(int(mask_w / 2), background_width - int(mask_w / 2))
                new_offset_y = randint(int(mask_h / 2), background_height - int(mask_h / 2))

            new_p1 = (new_offset_x - mask_w / 2, new_offset_y - mask_h / 2)
            new_p2 = (new_offset_x + mask_w / 2, new_offset_y + mask_h / 2)
            new_location = [new_p1, new_p2]

            if self.check_location(new_location, self.max_iou):
                offset_x = new_offset_x
                offset_y = new_offset_y

                self.locations.append(new_location)
                cv2.rectangle(
                    self.placement_map,
                    (offset_x - mask_w / 2, offset_y - mask_h / 2),
                    (offset_x + mask_w / 2, offset_y + mask_h / 2),
                    color=class_idx,
                    thickness=-1,
                )

                break

        return offset_x, offset_y
