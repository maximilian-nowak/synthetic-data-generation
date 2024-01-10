# Synthetic Data generation

This repository contains a comprehensive pipeline for generating synthetic data. The pipeline is designed to create somewhat realistic synthetic data with labeled annotations, offering flexibility in customization. It places objects into a scene.

# Installation

To get started first make sure you have **python3.9** installed.
```shell
pip install virtualenv
virtualenv
source
pip install -r requirements.txt
```

---

# Usage

### Preparation of the source images package
The pipeline requires a folder with objects and backgrounds as an input. The structure of the file should look like:

* *objects/*
	* class1/
		* *image1.jpg*
		* *image2.jpg*
			...
	* class2/
		* *image1.jpg*
		* *image2.jpg*
		* ...
	* other/
		* *image1.jpg*
		* *image2.jpg*
		* ...
* backgrounds/
	* *image1.jpg*
	* *image1_front.png*
	* *image1_placement-area.png*
	* *image2.jpg*
	* *image2_front.png*
	* *image2_placement-area.png*
	* ...


Directories names *class1, class2* etc. will be generated classes names.

'Other' is a class processed in the same procedure as others, but not included in annotations. For this class consider objects from the background, which are appearing often in a scene.

### How to add new product/background images

To add product images for synthetic data generation, you can follow the following steps:

1. Gathering Product Images:
    Identify the products you want to generate synthetic data for.
    Gather relevant object images on the internet using search engines or e-commerce websites, Use SegmentAnything or use photographs of objects. 
    Ideally these images already have a alpha channel.
2. Preparing Images with Jupyter Notebooks:
	Once you have gathered the product images, you can use the PrepareProductImages.ipynb Notebook for image preprocessing.
3. Putting Images into folder structure:
	Once you have prepared the product images, you can put them into above-mentioned structure.

To add background images for synthetic data generation, you can follow the following steps:

1. Gathering Fridge Background Images:
    Search for background images on the internet or in your dataset. Make sure to capture a variety of lighting conditions, and background settings to create diverse background images. Also make sure that there are empty spaces to insert objects into.
2. Preparing Images with Jupyter Notebooks:
   Once you have gathered the background images, you can use the PrepareBackgroundImages.ipynb Notebook for image preprocessing.
3. Using GIMP to Add a Mask for Placement Area and Foreground:
   After preparing the background images, you can use GIMP (GNU Image Manipulation Program) to add a mask for the placement area and foreground. 
   1. Open the background image in GIMP.
   2. Use GIMP's tools to create a mask that represents the placement area of the product images and the foreground (e.g., the fridge holding bar).
   3. Save the background images with the added mask as a PNG, to preserve the transparency information.
4. Putting Images into folder structure:
	Once you have prepared the background images, you can put them into above-mentioned structure.



### Preparation of the config file

* *generated_image_format* - format of the generated images, currently supported:  *"jpg"*,  *"png"*,  *"bmp"*
* *num_images* - number of images to be generated
* *min_size* - minimal relative size of the blended object in respect to background image (in percentage). If set to -1 or not present no scaling will be performed (i.e., the original object image size will be used)
* *max_size* - maximal relative size of the blended object in respect to background image (in percentage). If set to -1 or not present no scaling will be performed (i.e., the original object image size will be used)
* *min_num_obj* - minimal number of objects to be blended in single image. If set to or above 70, some images will not be generated if "none" cropper option is selected and an Opencv error will arise. If this case use please other cropper options. 
* *max_num_obj* - maximal number of objects to be blended in single image. If set to or above 70, some images will not be generated if "none" cropper option is selected and an Opencv error will arise. If this case use please other cropper options.
* *max_iou* - maximal value of intersection over union. The value should be in range [0.0 - 1.0], where 0.0 doesn't allow any obstructions.
* *data_augmentation_options* - image processing augmentations to be applied on every image before blending. Currently implemented: *"random_rotation", "increase_brightness", "random_blur"*.
* *use_3D_transformation* - apply 3D transformation on object images.
* *cropper* - algorithm used for cropping stage, currently implemented: *"salient", "selective_search", "simple"*. Default is *"salient"*.
* *blender* - algorithms used for blending stage, currently implemented: *"alpha", "poisson", "seamless", "pyramid","noblend"*. Default is *"seamless"*. You can use a list.
* *fit_object_brightness_to_scene* - the objects brightness will be fitted to the scene.
* *proportions* - controls the of the implanted objects size
* *multiprocessing* - true or false. If true it will use all the CPU cores to generate the images. If false or field not present, the single core computation will be used.
* *preset_path* - allows to set a folder with placement records to reproduce a generation.
* *ignored_saved_size* Toggle to ignore the size of objects placement records.
* *generate_placement_mask* - Generate in addition to the image a placement mask used for the Masked Image Enhancement.
* *use_pyramid_blender_for_reblending* - Use pyramid blending for the re-blending of the foreground.
* *use_placement_area* - Activating this requires a placement mask in the background images. Object will only be placed inside the allowed area.
* *raise_exceptions* - Should exception be raised or ignored.
* *random_seed* Set seed for image generation
* *label_format* label format. Currently implemented: *"voc", "yolo"*

Sample config file: *configs/config_detection.json*


### Running locally

```shell
python3 run.py --source samples/ --config configs/config_detection.yaml
```
