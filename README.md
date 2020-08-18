# Dense 3D Reconstruction from Stereo in ROS


This is a ROS package for real-time 3D reconstruction from stereo images using either [LIBELAS](http://www.cvlibs.net/software/libelas/) 
or stereo blockmatching from OpenCV , for generating dense disparity maps. The 3D point cloud is also generated and can be visualized on rviz.

- Author: [Pushyami Kaveti](http://pushyamikaveti.github.io/)

## Dependencies

- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
- [cmake](http://www.cmake.org/cmake/resources/software.html)
- [gflags](https://github.com/gflags/gflags)
- [Boost](http://www.boost.org/)
- [OpenCV == 3.3.1 ](https://github.com/opencv/opencv) 

## Stereo Calibration

A calibrated pair of cameras is required for stereo rectification and calibration files should be stored in a `.yml` file. 
[This repository](https://github.com/sourishg/stereo-calibration) contains all the tools and instructions to calibrate stereo cameras.

Please see a sample calibration file in the `calibration/` folder.

## Compiling

Clone the repository:

```bash
$ git clone https://github.com/umass-amrl/stereo_dense_reconstruction
```

For compiling the ROS package, `rosbuild` is used. Add the path of the ROS package to `ROS_PACKAGE_PATH` and put the following line in your `.bashrc` file. 
Replace `PATH` by the actual path where you have cloned the repository:

```bash
$ export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/PATH
```

Execute the `build.sh` script:

```bash
$ cd stereo_dense_reconstruction
$ chmod +x build.sh
$ ./build.sh
```

## Running Dense 3D Reconstruction

```bash
$ ./bin/dense_reconstruction [OPTION...]
```

```bash
Usage: dense_reconstruction [OPTION...]
  -l, --left_topic=STR       Left image topic name
  -r, --right_topic=STR      Right image topic name
  -c, --calib_file=STR       Stereo calibration file name
  -w, --calib_width=NUM      Calibration image width
  -h, --calib_height=NUM     Calibration image height
  -u, --out_width=NUM        Rectified image width
  -v, --out_height=NUM       Rectified image height
  -d, --debug=NUM            Set d=1 for cam to robot frame calibration
```

This node outputs the dense disparity map as a grayscale image on the topic `/camera/left/disparity_map` and the corresponding point cloud on the topic 
`/camera/left/point_cloud`.

A sample dataset can be found [here](https://greyhound.cs.umass.edu/owncloud/index.php/s/3g9AwCSkGi6LznK).

## Point Cloud Transformation

The point cloud can be viewed on `rviz` by running:

```bash
$ rosrun rviz rviz
```

To transform the point cloud to a different reference frame, the `XR` and `XT` matrices (rotation and translation) in the calibration file need to be changed. 
This can be done real-time by the running:

```bash
$ rosrun rqt_reconfigure rqt_reconfigure
```

If you change the Euler Angles in `rqt_reconfigure` you should be able to see the point cloud transform. Don't forget to set `d=1` when running the 
`dense_reconstruction` node. This prints out the new transformation matrices as you transform the point cloud.

## License

This software is released under the [GNU GPL v3 license](LICENSE).