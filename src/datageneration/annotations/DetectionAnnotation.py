"""
Module with Detection task that creates annotations

Classes:
    Detection
"""
import os.path
import xml.etree.ElementTree as et
from typing import List

import numpy as np


def create_annotation_voc(output_dir: str, idx: int, image: np.ndarray, detections: List):
    """
    Creates xml annotation for object detection task
    """
    results = {"results": detections}

    width = image.shape[1]
    height = image.shape[0]

    root = et.Element("annotation")
    size = et.SubElement(root, "size")
    et.SubElement(size, "width").text = str(width)
    et.SubElement(size, "height").text = str(height)
    et.SubElement(size, "depth").text = "3"

    # Include metadata fields if available
    metadata = et.SubElement(root, "metadata")
    for key, value in results.get("metadata", {}).items():
        et.SubElement(metadata, key).text = str(value)

    for detection in results["results"]:
        bbox_object = et.SubElement(root, "object")
        classname = detection["classname"]
        if classname == "other":  # Object from class other are getting ignored as labels
            continue
        et.SubElement(bbox_object, "name").text = classname

        if "confidence" in detection.keys():
            confidence = detection["confidence"]
            et.SubElement(bbox_object, "confidence").text = str(confidence)

        bndbox = et.SubElement(bbox_object, "bndbox")

        # COCO to VOC
        x1 = int(detection["bbox"][0])
        y1 = int(detection["bbox"][1])
        x2 = int(detection["bbox"][2]) + x1
        y2 = int(detection["bbox"][3]) + y1

        x1, x2 = [max(min(x, width), 0) for x in [x1, x2]]  # clip width in range [0, width]
        y1, y2 = [max(min(y, height), 0) for y in [y1, y2]]  # clip height in range [0, height]

        assert x1 != x2 and y1 != y2

        et.SubElement(bndbox, "xmin").text = str(x1)
        et.SubElement(bndbox, "ymin").text = str(y1)
        et.SubElement(bndbox, "xmax").text = str(x2)
        et.SubElement(bndbox, "ymax").text = str(y2)

    tree = et.ElementTree(root)
    et.indent(tree, space="\t", level=0)

    output_file_path = os.path.join(output_dir, "labels", str(idx) + ".txt")
    tree.write(output_file_path)


def create_annotation_yolo(output_dir: str, idx: int, image: np.ndarray, results: List):
    """
    Creates yolo annotation for object detection task

    writes format as: center_x , center_y , width, height ~ (all normalized)

    """
    img_width = image.shape[1]
    img_height = image.shape[0]

    label_file = []
    for bnd in results:
        cls = bnd['classname']
        xmin = bnd['bbox'][0]
        xmax = bnd['bbox'][1]
        ymin = bnd['bbox'][2]
        ymax = bnd['bbox'][3]

        width = xmax - xmin
        height = ymax - ymin

        xmin, xmax = [max(min(x, img_width), 0) for x in [xmin, xmax]]  # clip width in range [0, width]
        ymin, ymax = [max(min(y, img_height), 0) for y in [ymin, ymax]]  # clip height in range [0, height]

        assert xmin != xmax and ymin != ymax

        x1 = str((xmin + width / 2) / img_width)
        y1 = str((ymin + height / 2) / img_height)
        w = str(width / img_width)
        h = str(height / img_height)

        label_file.append(str(cls) + " " + x1 + " " + y1 + " " + w + " " + h + "\n")

    output_file_path = os.path.join(output_dir, "labels", str(idx) + ".txt")
    with open(output_file_path, 'w') as file:
        file.writelines(label_file)
