# Synthetic Data generation for traffic signs

Firstly, it is recommended to set up the project as described in readme.md.

## source images package

The resource for generating synthetic training data for traffic signs can be found in the folder `traffic_objects_package`.  

Structure:

Each folder represents a class label. Objects in the `other` folder will be used as well, but not be labelled.

* traffic_objects_package
    * *objects/*
        * 0/
            * *green-light.jpg*
        * 1/
            * *red-light.jpg*
        * 2/
            * *pit-in1.jpg*
            * ...
        * 3/
            * *pit-out1.jpg*
            * ...
        * 4/
            * *park-parallel1.jpg*
            * ...
        * 5/
            * *park-cross1.jpg*
            * ...
        * 6/
            * *overtaking-prohibited1.jpg*
            * ...
        * 7/
            * *overtaking-prohibited1.jpg*
            * ...
        * other/
            * *disruptive-object1.jpg*
            * *disruptive-object2.jpg*
            * ...
    * backgrounds/
        * *image1.jpg*
        * *image1_placement-area.png*
        * *image2.jpg*
        * *image2_placement-area.png*
        * ...

Before starting the process of generating a dataset, make sure that this folder only contains the objects and classes that you want to be included in your dataset (e.g. for a dataset containing only traffic lights, this folder should only list `0/` and `1/` - optionally `others` as well).

## configuration

The folder `configs` contains three different config files:

- config_trafficsigns.yaml: optimized for generating only traffic signs (no lights)
- config_traffilights.yaml: optimized for generating only traffic lights (no signs)
- config_traffix_mixed.yaml: optimized for generating all traffic objects

## generating datasets

python3 run.py --source traffic_objects_package/ --config configs/config_traffic_mixed.yaml


## helper scripts

After generating a dataset, it is advisable brush up the format and structure of those files.

1. Give each image/label pair a unique identifier:

    This is particularly useful when you plan on combining this dataset later on with training data from other sources. This way there won't be any name conflicts in the project.

    ```
    cd ./download/dataset-name
    python ../../scripts/uudify_frames.py 
    ```

2. Create a stats.txt:

    Add a short txt file to your dataset explaining the nature of the included data, e.g. a quick overview of all class objects could be useful in the future.

    ```
    cd ./download/dataset-name
    python ../../scripts/count_objects.py 
    ```
## Mask creation
Takes road images and gives masks back
    ```
    python3 mask_creation.py input_folder output_folder
    ```

## Statistics to WS23/24

For anyone interested in an overview of all our produced datasets is invited to have a look at `/docs/dataset_stats_ws2324`.
