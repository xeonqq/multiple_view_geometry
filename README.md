## Demo on how to calculate epipolar line
[![Build Status](https://travis-ci.com/xeonqq/multiple_view_geometry.svg?branch=master)](https://travis-ci.com/xeonqq/multiple_view_geometry) [![Total alerts](https://img.shields.io/lgtm/alerts/g/xeonqq/multiple_view_geometry.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xeonqq/multiple_view_geometry/alerts/)

### Given: 
* The two camera positions in world frame, 
* The position of the cube in world frame, 
* The camera intrinsics

### Then:
* Calculate the projection of the keypoints from the cube onto the image frame
* Calculate the epipolar line in each image using its coorespondence and essential matrix

### Demo
![](imgs/epipolar_line.gif)

### Command to run the demo:
```bash
python ./epipolar_geometry.py
```
or interact with the notebook
```bash
jupyter notebook epipolar_geometry.ipynb
```
