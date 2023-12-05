"""
Module for generator for object detection tasks

Classes:
    DetectionGenerator
"""

import os
import pickle
from random import choice, randint
from typing import Any, Dict, List, Optional, Tuple, Union

from src.datageneration.annotations.DetectionAnnotation import create_annotation_yolo, create_annotation_voc
from src.datageneration.augmentation.augmentations import apply_augmentations
from src.datageneration.blender.abstract_blender import fit_brightness_to_scene, restore_foreground
from src.datageneration.blender.blenders import available_blenders
from src.datageneration.cropper.croppers import get_bounding_box_with_closing, available_croppers, extract_mask, \
    limit_size_of_obj
from src.datageneration.files import get_angled_image_from_folder, get_backgrounds

import cv2
import numpy as np

from src.datageneration.scripts.image_processing import resize
from src.datageneration.scripts.placement import IoUCheckingPlacer
from src.datageneration.scripts.proportions import available_proportions_computator
from src.datageneration.scripts.transformer import apply_3d_transformation

ConfigType = Dict[str, Optional[Union[bool, str, int, float, Dict[str, Any], List[str]]]]


class DetectionGenerator:
    """
    Class of object detection images generator

    Methods:
        _pick_instances
        _generate_image
    """

    def __init__(self, config: ConfigType):

        self.raise_exceptions = config["raise_exceptions"]
        self.generate_placement_mask = config["generate_placement_mask"]
        self.ignored_saved_size = config["ignored_saved_size"]

        self.output_dir = config["output_dir"]
        self.generated_image_format = config["generated_image_format"]
        self.label_format = config["label_format"]

        self.use_placement_area = config["use_placement_area"]
        self.use_3D_transformation = config["use_3D_transformation"]
        self.reset_foreground = config["reset_foreground"]
        self.fit_object_brightness_to_scene = config["fit_object_brightness_to_scene"]
        self.use_pyramid_blender_for_reblending = config["use_pyramid_blender_for_reblending"]

        self.min_size = config["min_size"]
        self.max_size = config["max_size"]

        self.cropper = available_croppers[config["cropper"]]
        self.proportions_computator = available_proportions_computator[config["proportions"]]
        self.placer = IoUCheckingPlacer(max_iou=config["max_iou"])
        self.blender = []
        for b in config["blender"]:
            self.blender.append(available_blenders[b])
        self.data_augmentation_options = []
        for a in config["data_augmentation_options"]:
            self.data_augmentation_options.append(a)

        os.makedirs(os.path.join(self.output_dir, "images"))
        os.makedirs(os.path.join(self.output_dir, "labels"))
        os.makedirs(os.path.join(self.output_dir, "placement_records"))
        if config["generate_placement_mask"]:
            os.makedirs(os.path.join(self.output_dir, "placement_mask"))

    def generate_image(self, job: tuple):
        """
        Generates single image
        """

        idx = job[0]
        background_path = job[1]
        object_name_pool = job[2]

        detections = []
        placement_record = {
            "idx": idx,
            "background_path": os.path.basename(background_path),
            "placed_elements": [],
        }

        background, allowed_areas_mask, foreground, background_config = get_backgrounds(background_path,
                                                                                        self.use_placement_area,
                                                                                        self.reset_foreground)
        background_ori = background.copy()
        placement_masks = []

        implantation_counter = 0
        for class_name, object_path in object_name_pool:
            try:
                obj_size = randint(self.min_size, self.max_size)

                if os.path.isdir(object_path):
                    object_path = get_angled_image_from_folder(object_path, background_config)

                object_image = cv2.imread(object_path)
                mask = extract_mask(object_path, self.cropper)
                object_image, mask = limit_size_of_obj(object_image, mask)

                proportion = self.proportions_computator.compute_proportion(
                    mask, background, obj_size
                )
                if "_half" in object_path:
                    proportion *= 0.70
                mask = resize(mask, proportion)
                object_image = resize(object_image, proportion)
                object_image, mask = apply_augmentations(object_image, mask, self.data_augmentation_options)

                offset_y, offset_x = self.placer.compute_offsets(
                    mask,
                    background,
                    allowed_areas_mask,
                    self.use_placement_area,
                )

                if self.use_3D_transformation:
                    object_image, mask, (offset_x, offset_y) = apply_3d_transformation(
                        object_image, mask, background_config, (offset_x, offset_y)
                    )

                if offset_y > 0 and offset_x > 0:
                    if class_name not in ("negative", "other"):
                        detections.append(
                            self._get_detection(mask, class_name, (offset_x, offset_y))
                        )
                        placement_masks.append((mask, (offset_x, offset_y)))

                    if self.fit_object_brightness_to_scene:
                        object_image = fit_brightness_to_scene(object_image, background_ori, mask)

                    blender = choice(self.blender)
                    background = blender.blend_object(
                        background, object_image, mask, (offset_x, offset_y)
                    )

                    if background is None:
                        if self.raise_exceptions:
                            raise ValueError("ERROR! - with:", object_path, 'and', background_path)
                    else:
                        # restore foreground if present
                        if foreground is not None and self.reset_foreground:
                            background = restore_foreground(
                                background,
                                foreground,
                                background_ori,
                                self.use_pyramid_blender_for_reblending,
                            )

                        object_name = os.path.basename(object_path)
                        placed_elements = {
                            "object_name": object_name,
                            "offset_y": offset_y,
                            "offset_x": offset_x,
                            "size": obj_size,
                            "implantation_counter": implantation_counter,
                        }
                        placement_record["placed_elements"].append(placed_elements)
                        implantation_counter += 1

            except cv2.error as e:
                print("ERROR!", e, "- with:", object_path, 'and', background_path)
                if self.raise_exceptions:
                    raise e
            except ValueError as e:
                print("ERROR!", e, "- with:", object_path, 'and', background_path)
                if self.raise_exceptions:
                    raise e
            except Exception as e:
                print("ERROR!", e, "- with:", object_path, 'and', background_path)
                if self.raise_exceptions:
                    raise e

        if len(detections) > 0:
            self.finalize(
                placement_record,
                detections,
                background,
                placement_masks,
                foreground,
                idx,
            )

    def finalize(
        self,
        placement_record: dict,
        detections: List,
        background: np.array,
        placement_masks: np.array,
        foreground: np.array,
        idx: int,
    ):
        """
        finalizes image construction
        Args:
            foreground:
            placement_record: Dict
            detections: List
            background: np. array
            placement_masks: np.array
            idx: int

        Returns: None

        """
        # Write placement_mask
        if self.generate_placement_mask:
            placement_mask = self._generate_placement_mask(background, placement_masks, foreground)
            placement_mask_filename = (str(idx) + "_placement_mask" + "." + self.generated_image_format)
            placement_mask_path = os.path.join(self.output_dir, "placement_mask", placement_mask_filename)
            cv2.imwrite(placement_mask_path, placement_mask)

        if self.label_format == "yolo":
            create_annotation_yolo(self.output_dir, idx, background, detections)
        elif self.label_format == "voc":
            create_annotation_voc(self.output_dir, idx, background, detections)
        else:
            raise ValueError("Annotation format is invalid:", self.label_format)

        # Write final image
        gen_filename = str(idx) + "." + self.generated_image_format
        cv2.imwrite(os.path.join(self.output_dir, "images", gen_filename), background)

        # Write placement_record
        placement_record_filename = str(idx) + ".placement_record"
        with open(os.path.join(self.output_dir, "placement_records", placement_record_filename), "wb") as file:
            pickle.dump(placement_record, file)

    def generate_image_from_record(self, idx, placed_elements, background_path):
        """
        Generates single image from record
        """
        detections = []
        placement_masks = []

        background, _, foreground, background_config = get_backgrounds(background_path,
                                                                       self.use_placement_area,
                                                                       self.reset_foreground)
        background_ori = background.copy()

        placement_record = {
            "idx": idx,
            "background_path": os.path.basename(background_path),
            "placed_elements": [],
        }

        for placed_object in placed_elements:

            object_path = placed_object["object_name"]
            object_class = os.path.basename(os.path.dirname(object_path))
            offset_x = placed_object["offset_x"]
            offset_y = placed_object["offset_y"]
            size = placed_object["size"]
            if self.ignored_saved_size:
                size = randint(self.min_size, self.max_size)

            try:
                object_image = cv2.imread(object_path)
                mask = extract_mask(object_path, self.cropper)
                object_image, mask = limit_size_of_obj(object_image, mask)

                proportion = self.proportions_computator.compute_proportion(
                    mask, background, size
                )
                if "_half" in object_path:
                    proportion *= 0.80
                mask = resize(mask, proportion)
                object_image = resize(object_image, proportion)
                object_image, mask = apply_augmentations(object_image, mask, self.data_augmentation_options)

                if self.use_3D_transformation:
                    object_image, mask, _ = apply_3d_transformation(
                        object_image, mask, background_config, (offset_x, offset_y)
                    )

                if object_class not in ("negative", "other"):
                    detections.append(self._get_detection(mask, object_class, (offset_x, offset_y)))
                    placement_masks.append((mask, (offset_x, offset_y)))

                if self.fit_object_brightness_to_scene:
                    object_image = fit_brightness_to_scene(object_image, background_ori, mask)

                blender = choice(self.blender)
                background = blender.blend_object(
                    background, object_image, mask, (offset_x, offset_y)
                )

                # restore foreground if present and active
                if foreground is not None and self.reset_foreground:
                    background = restore_foreground(
                        background,
                        foreground,
                        background_ori,
                        self.use_pyramid_blender_for_reblending
                    )

            except cv2.error as e:
                print("ERROR!", e, "- with:", object_path)
                if self.raise_exceptions:
                    raise e
            except ValueError as e:
                print("ERROR!", e, "- with:", object_path)
                if self.raise_exceptions:
                    raise e
            except Exception as e:
                print("ERROR!", e, "- with:", object_path)
                if self.raise_exceptions:
                    raise e

        self.finalize(
            placement_record,
            detections,
            background,
            placement_masks,
            foreground,
            idx,
        )

    @staticmethod
    def _generate_placement_mask(background, placement_masks, foreground):
        """
        Generates a placement mask combined of all mask
        """
        placement_mask = np.zeros(background.shape[:2], dtype=np.uint8)

        for mask, offset in placement_masks:
            p1, p2 = get_bounding_box_with_closing(mask)
            object_width = (p2[0] - p1[0]) / 2
            object_height = (p2[1] - p1[1]) / 2
            shift_x = offset[0] - object_width
            shift_y = offset[1] - object_height
            m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            mask_extended = cv2.warpAffine(
                mask[p1[1]: p2[1], p1[0]: p2[0]],
                m,
                (background.shape[1], background.shape[0]),
            )
            placement_mask[mask_extended > 0] = 255

        # Remove foreground from placement mask
        if foreground is not None:
            alpha_foreground = foreground[:, :, 3] / 255.0
            kernel = np.ones((3, 3), np.uint8)
            alpha_foreground = cv2.erode(alpha_foreground, kernel, iterations=1)
            placement_mask = placement_mask * (1 - alpha_foreground)

        return placement_mask

    @staticmethod
    def _get_detection(mask: np.ndarray, class_name: str, offset: Tuple[int, int]) -> Dict[str, Union[str, List[int]]]:
        """
        Creates generated detection annotation

        Args:
            mask: mask for object bounding box getter
            class_name: name of class that will be extracted
            offset: offset where augmented bounding box will be placed

        Returns:
            Bounding box results with class name and bounding box coordinates
        """
        p1, p2 = get_bounding_box_with_closing(mask)
        half_object_width = (p2[0] - p1[0]) // 2
        half_object_height = (p2[1] - p1[1]) // 2

        assert half_object_height * half_object_width > 100

        x1 = offset[0] - half_object_width
        y1 = offset[1] - half_object_height
        x2 = offset[0] + half_object_width
        y2 = offset[1] + half_object_height
        results = {"classname": class_name, "bbox": [x1, y1, x2 - x1, y2 - y1]}

        return results



