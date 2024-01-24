# Synthetic Data generation for traffic signs

Firstly, it is recommended to set up the project as described in readme.md.

### source images package

The resource for generating synthetic training data for traffic signs can be found in the folder `traffic_objects_package`.  

Structure:

Each folder represents a class as label. Objects in the other folder will be used as well, but not be labelled.

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

## Generating datasets

python3 run.py --source data/ --config configs/config_traffic_mixed.yaml