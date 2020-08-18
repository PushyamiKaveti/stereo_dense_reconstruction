//
// Created by auv on 7/11/20.
//
#include "elas.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/PointCloud2.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"


#include "utils.h"
#include "DepthReconstructor.h"

using namespace cv;
using namespace std;

//Define the arguments to the program
DEFINE_string(left_topic, "/left/image_raw", "Left image topic name");
DEFINE_string(right_topic, "/right/image_raw", "Right image topic name");
DEFINE_string(calib_file, "calibration.yaml", "Stereo calibration file name");

DEFINE_int32(calib_width, 640, "Calibration image width");
DEFINE_int32(calib_height, 480, "Calibration image height");

DEFINE_bool(debug, false, "Debug Mode");
DEFINE_bool(is_ros, true, "publish the depth to ROS or save images on disk");
DEFINE_int32(algo, 1, "Which stereo alhorithm to run ELAS =1 , SGBM=2");
DEFINE_string(output_dir, ".", "directory where the rgb and depth images should be saved");
DEFINE_bool(color, false, "Should the images be saved in color or mono");

Size out_img_size;
Size calib_img_size;
FileStorage calib_file;

vector<Mat> K_mats , D_mats, R_mats, T_mats, P_mats;
image_transport::Publisher dmap_pub;
ros::Publisher pcl_pub;

DepthReconstructor* depthProc;
string write_folder = ".";



void imgCallback(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right) {

        Mat tmpL = cv_bridge::toCvShare(msg_left, "mono8")->image;
        Mat tmpR = cv_bridge::toCvShare(msg_right, "mono8")->image;
        Mat tmpColor;
        if (FLAGS_color){
            tmpColor = cv_bridge::toCvShare(msg_left, "bgr8")->image;
        }

        if (tmpL.empty() || tmpR.empty())
            return;

        Mat disp, vdisp, depthMap;
        depthProc->calcDisparity(tmpL, tmpR, disp, depthMap);
        cv::normalize(disp, vdisp, 0, 256, cv::NORM_MINMAX, CV_8U);

        if(FLAGS_debug){
            Mat pair;
            pair.create(calib_img_size.height, calib_img_size.width * 2, CV_8U);
            depthProc->img1_rect.copyTo(pair.colRange(0, calib_img_size.width));
            depthProc->img2_rect.copyTo(
                    pair.colRange(calib_img_size.width, calib_img_size.width * 2));
            for (int j = 0; j < calib_img_size.height; j += 50)
                cv::line(pair, cv::Point(0, j), cv::Point(calib_img_size.width * 2, j), cv::Scalar(255));
            cv::imshow("rectified", pair);
            cv::waitKey(0);
        }


        //sensor_msgs::PointCloud2Ptr points_msg = boost::make_shared<sensor_msgs::PointCloud2>();
        //Mat depth;
        //depthProc->convertToDepthMap(disp, depth);
        Mat img_left_color;
        if(FLAGS_color){
            cv::remap(tmpColor, img_left_color, depthProc->rectMapLeft_x, depthProc->rectMapLeft_y, cv::INTER_LINEAR);
        }
        else{
            cvtColor(depthProc->img1_rect, img_left_color, CV_GRAY2BGR);
        }
        if(!FLAGS_is_ros){
            std::string depth_path = write_folder+"/depth/" + to_string(msg_left->header.stamp.sec)+ "." + to_string(msg_left->header.stamp.nsec) + ".png";
            if (boost::filesystem::exists(depth_path))
                return;
            ROS_INFO_STREAM(std::to_string(msg_left->header.seq));
            ROS_INFO_STREAM(msg_left->header.stamp.sec <<"  "<<msg_left->header.stamp.nsec);
            ROS_INFO_STREAM(msg_right->header.stamp.sec<<"  "<<msg_right->header.stamp.nsec);
            //depth_path.append();
            std::string rgb_path;
            rgb_path.append(write_folder+ "/rgb/"+to_string(msg_left->header.stamp.sec) + "." + to_string(msg_left->header.stamp.nsec) + ".png");


            cv::imwrite(depth_path , depthMap);
            cv::imwrite(rgb_path, img_left_color);
        }
        else{
            //pcl_pub.publish(points_msg);

            sensor_msgs::ImagePtr disp_msg;
            disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", vdisp).toImageMsg();
            dmap_pub.publish(disp_msg);
        }


        waitKey(30);

}



int main(int argc, char** argv){

    const char* calib_file_name;
    const char* left_img_topic;
    const char* right_img_topic;

    // parse arguments
    google::ParseCommandLineFlags(&argc, &argv, true);

    calib_img_size = Size(FLAGS_calib_width, FLAGS_calib_height);

    calib_file = FileStorage(FLAGS_calib_file, FileStorage::READ);
    cout<<FLAGS_calib_file<<endl;
    // read all the matrices K, D, R, T
    read_calib_params(calib_file,K_mats,D_mats,R_mats,T_mats,P_mats);

    //create stereo object
    depthProc = new DepthReconstructor(FLAGS_algo,  calib_img_size, true, FLAGS_debug);
    depthProc->init(K_mats[0] , D_mats[0], K_mats[1], D_mats[1],R_mats[1], T_mats[1]);
    write_folder = FLAGS_output_dir;

   // if (FLAGS_is_ros){
        ros::init(argc, argv, "stereo_reconstruction");
        ros::NodeHandle nh;
        image_transport::ImageTransport it(nh);

        message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, FLAGS_left_topic, 1);
        message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, FLAGS_right_topic, 1);

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
        message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(1), sub_img_left, sub_img_right);
        sync.registerCallback(boost::bind(&imgCallback, _1, _2));


        dmap_pub = it.advertise("/camera/left/disparity_map", 1);
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/camera/left/point_cloud",1);

        ros::spin();

    //}


    return 0;
}