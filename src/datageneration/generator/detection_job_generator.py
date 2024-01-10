"""
Abstract generator class for all generation module

Classes:
    AbstractGenerator
"""

import multiprocessing
import concurrent.futures
import random
import sys
from random import choice
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from src.datageneration.generator.detection_generator import DetectionGenerator

ConfigType = Dict[str, Optional[Union[str, int, float, Dict[str, Any], List[str]]]]


class DetectionJobGenerator:

    def __init__(self, config: ConfigType, image_generator: DetectionGenerator):
        self.config = config
        self.image_generator = image_generator

        random.seed(config["random_seed"])

    def create_job_list(self, objects_images: Dict[str, List[str]], background_images: List) -> List[Tuple]:
        """
        Creates job list for generator
        """
        joblist = []
        for i in tqdm(range(self.config["num_images"]), desc="Create job list"):
            choice_background_image = choice(background_images)

            choice_objects = []
            number_of_obj_per_scene = random.randint(self.config["min_num_obj"], self.config["max_num_obj"])
            for _ in range(number_of_obj_per_scene):
                choice_class = choice(list(objects_images.keys()))
                choice_object_image = choice(objects_images[choice_class])
                choice_objects.append((choice_class, choice_object_image))

            joblist.append((i, choice_background_image, choice_objects))
        return joblist

    def generate_data_single(self, joblist: List[Tuple]):
        """
        Generates data with single thread
        """
        for job in tqdm(joblist, desc="Generating images"):
            self.generate_data(job)

    def generate_data_multi(self, joblist: List[Tuple]):
        """
        Generates data with multi thread approach
        """
        number_of_cores = multiprocessing.cpu_count()
        print("----------------------")
        print("Number of cores available:", number_of_cores)
        print("----------------------")
        pool = multiprocessing.Pool(processes=number_of_cores)
        for _ in tqdm(pool.imap_unordered(self.generate_data, joblist), total=len(joblist), desc="Generating images"):
            pass

    def generate_data(self, job: Tuple):
        """
        Function for multiprocess data generation
        """
        try:
            self.image_generator.generate_image(job)
        except AttributeError:
            print(f"Image with background {job} could not be generated!")
            if self.config["raise_exceptions"]:
                raise AttributeError("Image with background {job} could not be generated!")

    def generate_data_single_from_records(self, job_list):
        """
        Generates data with single process from records
        """
        for job in tqdm(job_list, desc="Generating images from records"):
            self.generate_data_from_record(job)

    def generate_data_multi_from_records( self, job_list):
        """
        Generates data with single process from records with multithreading
        """
        number_of_cores = multiprocessing.cpu_count()
        print("----------------------")
        print("Number of cores available:", number_of_cores)
        print("----------------------")
        pool = multiprocessing.Pool(processes=number_of_cores)
        for _ in tqdm(pool.imap_unordered(self.generate_data_from_record, job_list), total=len(job_list), desc="Generating images"):
            pass

    def generate_data_from_record(self, job: Dict):
        self.image_generator.generate_image_from_record(job["idx"], job["placed_elements"], job["background_path"])
