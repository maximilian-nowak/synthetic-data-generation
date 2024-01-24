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

Before starting the process of generating a dataset, make sure that this folder only contains the objects and classes that you want to include in your dataset (e.g. for a dataset containing only traffic lights, this folder should only list `0/` and `1/` - optionally `others` as well).

## configuration

The folder `config` contains three different config files:

- config_trafficsigns.yaml: optimized for generating only traffic signs (no lights)
- config_traffilights.yaml: optimized for generating only traffic lights (no signs)
- config_traffix_mixed.yaml: optimized for generating all traffic objects

## Generating datasets

python3 run.py --source traffic_objects_package/ --config configs/config_traffic_mixed.yaml