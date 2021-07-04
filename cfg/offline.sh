#! /bin/bash
./build.sh && ./bin/stereo_offline_bag \
--left_topic=/frl_uas7/camera_array/cam0/image_raw \
--right_topic=/frl_uas7/camera_array/cam1/image_raw \
--left_rect_topic=/frl_uas7/camera_array/cam0/image_rect \
--right_rect_topic=/frl_uas7/camera_array/cam1/image_rect \
--disparity_topic=/frl_uas7/camera_array/cam1/disparity \
--calib_width=1280 \
--calib_height=1024 \
--input_bag=/media/mithun/easystore/rosbags/2_uav_filtered.bag \
--calibrated=true \
--write_src_imgs=false \
--write_rect_imgs=false \
--write_disparity_img=false \
--publish2ros=true \
--debug=false \
--visualize=false \
--approx_policy=false \
--disparity_colormap=true \
--calib_file=/home/mithun/apps/stereo_dense_reconstruction/calibration/uav7.yaml