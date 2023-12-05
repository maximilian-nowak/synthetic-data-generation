"""
Script that can be run to get artificial data for training

Functions:
    parse_request_config
    generate_training_data
    pipeline
"""
import glob
import pickle

import yaml
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.datageneration.blender.blenders import available_blenders
from src.datageneration.cropper.croppers import available_croppers
from src.datageneration.files import get_object_images, get_background_images, SUPPORTED_IMAGE_EXTENSIONS
from src.datageneration.generator.detection_generator import DetectionGenerator
from src.datageneration.generator.detection_job_generator import DetectionJobGenerator

ConfigType = Dict[str, Optional[Union[str, int, float, Dict[str, Any], List[str]]]]


def parse_and_check_config(config: ConfigType) -> ConfigType:
    """
    Parses configuration dictionary of generation process. Checks for possible errors and adjust
    some fields if they occur empty in config

    Args:
        config: Configuration json read from file

    Returns:
        Parsed configuration dictionary
    """

    if "num_images" not in config:
        raise KeyError("Specify a number of images \n")

    if "generated_image_format" not in config:
        config["generated_image_format"] = "jpg"
    elif config["generated_image_format"] not in SUPPORTED_IMAGE_EXTENSIONS:
        message = (
            f"Unknown image format '{config['generated_image_format']}'. "
            f"Use one of: {SUPPORTED_IMAGE_EXTENSIONS} \n"
        )
        raise ValueError(message)

    if "min_num_obj" not in config:
        message = "Specify minimum number of objects in the config (min_num_obj) \n"
        raise ValueError(message)
    try:
        config["min_num_obj"] = int(config["min_num_obj"])
    except ValueError as e:
        message = "Invalid value of 'min_num_obj'!\n"
        raise ValueError(message) from e

    if "max_num_obj" not in config:
        message = "Specify maximum number of objects in the config (max_num_obj) \n"
        raise KeyError(message)
    try:
        config["max_num_obj"] = int(config["max_num_obj"])
    except ValueError as e:
        message = "Invalid value of 'max_num_obj'!\n"
        raise ValueError(message) from e

    if "min_size" not in config:
        config["min_size"] = -1
    else:
        try:
            config["min_size"] = int(config["min_size"])
        except ValueError as e:
            message = "Invalid value of 'min_size'!\n"
            raise ValueError(message) from e

    if "max_size" not in config:
        config["max_size"] = -1
    else:
        try:
            config["max_size"] = int(config["max_size"])
        except ValueError as e:
            message = "Invalid value of 'max_size'!\n"
            raise ValueError(message) from e

    if "max_iou" not in config:
        config["max_iou"] = 0.0
    else:
        config["max_iou"] = float(config["max_iou"])
        if 0.0 < config["max_iou"] > 1.0:
            raise ValueError("Invalid value of 'max_iou'! Use float between 0.0 and 1.0.\n")

    if "blender" not in config:
        config["blender"] = "stamping"
    else:
        for b in config["blender"]:
            if b not in available_blenders:
                message = f"Unknown blender '{config['blender']}'. Use one of: {available_blenders.keys()} \n"
                raise ValueError(message)

    if "cropper" not in config:
        config["cropper"] = "none"
    elif config["cropper"] not in available_croppers:
        message = f"Unknown cropper '{config['cropper']}'. Use one of: {available_croppers.keys()} \n"
        raise ValueError(message)

    #if "data_augmentation_options" not in config:
    #    config["data_augmentation_options"] = []
    #else:
    #    try:
    #        config["data_augmentation_options"] = list(config["data_augmentation_options"])
    #    except ValueError as e:
    #        message = "Invalid value of 'data_augmentation_options'!\n"
    #        raise ValueError(message) from e

    #    for data_augmentation_option in config["data_augmentation_options"]:
    #        if data_augmentation_option not in augmentations:
    #            message = (
    #                f"Unknown data augmentation option '{data_augmentation_option}'. "
    #                f"Use one of: {augmentations.keys()} \n"
    #            )
    #            raise ValueError(message)

    return config


def pipeline_prepare_source(source_full_path: Path, config_full_path: Path, target_dir: Path) -> (Dict, List):
    """
    prepare image sources
    """
    with open(config_full_path, 'r') as file:
        config = yaml.safe_load(file)

    parse_and_check_config(config)

    config["output_dir"] = target_dir

    objects_dir = source_full_path / "objects"
    backgrounds_dir = source_full_path / "backgrounds"

    shutil.copy(config_full_path, target_dir / "config.yaml")

    objects_images = get_object_images(str(objects_dir))
    if not objects_images:
        raise ValueError("No objects images!\n")
    print("Number of object-images:")
    for k, v in objects_images.items():
        print("-", k + ":", len(v))

    background_images = get_background_images(str(backgrounds_dir))
    print("Number of background-images:", len(background_images))
    if not background_images:
        raise ValueError("No backgrounds images!\n")

    # TODO: Check image integrity
    return config, objects_images, background_images


def pipeline_start(config: Dict, target_dir: Path, objects_images: Dict, background_images: List):

    image_generator = DetectionGenerator(config)
    job_generator = DetectionJobGenerator(config, image_generator)

    if config["preset_path"] != "":
        placements_records = load_placement_records(config)

        job_list = match_placement_records_with_data(objects_images, background_images, placements_records)

        if "multiprocessing" in config and config["multiprocessing"]:
            job_generator.generate_data_multi_from_records(job_list)
        else:
            job_generator.generate_data_single_from_records(job_list)
    else:
        job_list = job_generator.create_job_list(objects_images, background_images)

        if "multiprocessing" in config and config["multiprocessing"]:
            job_generator.generate_data_multi(job_list)
        else:
            job_generator.generate_data_single(job_list)


def load_placement_records(config) -> List:
    if not os.path.exists(config["preset_path"]):
        raise ValueError("preset_path does not exist")

    print("---- Using placement records from:", config["preset_path"], "----")

    placements_records = []
    for file in glob.glob(os.path.join(config["preset_path"], "*.placement_record")):
        with open(file, "rb") as filehandler:
            object_file = pickle.load(filehandler)
            placements_records.append(object_file)

    if not placements_records:
        raise ValueError("Placements_records directory is empty or incorrectly set!")

    print("---- regenerate:", len(placements_records), "images ----")
    return placements_records


def match_placement_records_with_data(objects_images, background_images, placements_records) -> List:

    matched_placements_records = []

    for placement_record in tqdm(placements_records, desc="Match placement records with source data"):
        matched_placement_record = {"idx": placement_record["idx"]}

        background_path = [
            i for i in background_images if i.endswith(placement_record["background_path"])
        ][0]
        matched_placement_record["background_path"] = background_path

        object_list = []
        dir_name = os.path.dirname(list(objects_images.values())[0][0])
        for obj in placement_record["placed_elements"]:
            placed_elements = {
                "object_name": os.path.join(dir_name, obj["object_name"]),
                "offset_y": obj["offset_y"],
                "offset_x": obj["offset_x"],
                "size": obj["size"],
                "implantation_counter": obj["implantation_counter"],
            }
            object_list.append(placed_elements)

        matched_placement_record["placed_elements"] = object_list

        matched_placements_records.append(matched_placement_record)
    return matched_placements_records


def pipeline_finalize(config: Dict, target_dir: Path, objects_images: Dict, background_images: List):
    df = create_dataset_csv(str(target_dir))
    df.to_csv(os.path.join(target_dir, 'dataset.csv'), sep='\t')


def create_dataset_csv(path):
    img_dir = os.path.join(path, 'images')
    df = pd.DataFrame(columns=['picture_url', 'label_url'])
    df['picture_url'] = [os.path.join('images', file) for file in os.listdir(img_dir)]
    df['label_url'] = [os.path.join('labels', file.replace('.jpg', '.xml')) for file in os.listdir(img_dir)]
    df['image_name'] = df.apply(lambda row: os.path.basename(row['picture_url']), axis=1)
    return df


def pipeline(opt, cwd):
    """
    Pipeline that runs image generation process
    """
    target_path = Path(cwd).joinpath(Path(opt.target))

    shutil.rmtree(target_path, ignore_errors=True)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    temp_dir = tempfile.mkdtemp(dir=str(target_path))

    print("----------------------")
    print("Result folder: ", temp_dir)
    print("----------------------")

    source_path = Path(cwd).joinpath(Path(opt.source))
    config_path = Path(cwd).joinpath(Path(opt.config))

    config, objects_images, background_images = pipeline_prepare_source(source_path, config_path, Path(temp_dir))

    pipeline_start(config, Path(temp_dir), objects_images, background_images)

    pipeline_finalize(config, Path(temp_dir), objects_images, background_images)





