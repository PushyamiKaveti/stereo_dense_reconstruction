# Dense 3D Reconstruction from Stereo in ROS


This is a ROS package for real-time 3D reconstruction from stereo images using either [LIBELAS](http://www.cvlibs.net/software/libelas/) 
or [stereo block matching](https://docs.opencv.org/3.4/d9/dba/classcv_1_1StereoBM.html) from OpenCV , for generating dense disparity maps. The 3D point cloud is also generated and can be visualized on rviz.

- Author: [Pushyami Kaveti](http://pushyamikaveti.github.io/)

## Dependencies

- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
- [cmake](http://www.cmake.org/cmake/resources/software.html)
- [gflags](https://github.com/gflags/gflags)
- [Boost](http://www.boost.org/)
- [OpenCV == 3.3.1 ](https://github.com/opencv/opencv) 

## Stereo Calibration

The intrinsic and extrinsic calibration parameters of a pair of cameras should be stored in a `.yml` file. A stereo pair can be calibrated
using a checkerboard rig using [openCV](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) or 
[Kalibr](https://github.com/ethz-asl/kalibr). A sample calibration file is provided in the calibration folder

## Compiling

Clone the repository of the ROS package in src/ folder of a catkin workspace :

```bash
$cd catkin_ws/src
$ git clone  https://github.com/PushyamiKaveti/stereo_dense_reconstruction.git
```

and execute 
```bash
$cd catkin_ws 
$catkin_make 
``` 


## Running the stereo app

```bash
$ ./bin/stereo_app [OPTION...]
```

```bash
Usage: dense_reconstruction [OPTION...]
   --left_topic=STR       Left image topic name
   --right_topic=STR      Right image topic name
   --calib_file=STR       Stereo calibration file name
   --calib_width=NUM      Calibration image width
   --calib_height=NUM     Calibration image height
   --is_ros=bool          is the dense reconstruction info published to ROS or saved to disk.
   --algo=NUM             which stereo algorithm to use ELAS=1, SGBM=2    
   --output_dir=STR       directory where the rgb and depth images should be saved      
   --color                for color or mono images
   --debug=bool           Set true for debug mode
```
if is_ros flag is true dense disparity grayscale image is published on the topic `/camera/left/disparity_map` and the corresponding point cloud on the topic 
`/camera/left/point_cloud`. Otherwise the depth map is saved onto disk under output_dir/depth folder. The depth map written to disk is not 
the disparity, but the actual depth image scaled by 5000.

## License

This software is released under the [GNU GPL v3 license](LICENSE).