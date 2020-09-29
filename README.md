## Multi view geometry key concepts in Python
[![Build Status](https://travis-ci.com/xeonqq/multiple_view_geometry.svg?branch=master)](https://travis-ci.com/xeonqq/multiple_view_geometry) [![Total alerts](https://img.shields.io/lgtm/alerts/g/xeonqq/multiple_view_geometry.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xeonqq/multiple_view_geometry/alerts/)

### 1. Demo on how to calculate epipolar line
#### Given: 
* The two camera positions in world frame, 
* The position of the cube in world frame, 
* The camera intrinsics

#### Then:
* Calculate the projection of the keypoints from the cube onto the image frame
* Calculate the epipolar line in each image using its coorespondence and essential matrix

#### Demo
![](imgs/epipolar_line.gif)

### Command to run the demo:
```bash
python ./epipolar_geometry.py
```
or interact with the notebook
```bash
jupyter notebook epipolar_geometry.ipynb
```

### 2. Eight points algorithm 
#### Given
 * The positions of 8 or more pairs of point correspondence in two frames
 * The camera intrinsics
#### Then:
 * calculate the essential matrix, and derive the translation and rotation of the two camera poses. [](tests/test_eight_point_algorithm.py)
