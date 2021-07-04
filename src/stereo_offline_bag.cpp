#include "elas.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/simple_filter.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sys/stat.h>

#include "utils.h"
#include "DepthReconstructor.h"
#include "depth_uncalibrated.h"

using namespace cv;
using namespace std;

//Define the arguments to the program
DEFINE_string(left_topic, "/left/image_raw", "Left image topic name");
DEFINE_string(right_topic, "/right/image_raw", "Right image topic name");
DEFINE_string(disparity_topic, "/left/disparity_map","disparity topic name");
DEFINE_string(pc2_topic, "/left/point_cloud", "point_cloud topic name");
DEFINE_string(left_rect_topic, "/left/image_rect","disparity topic name");
DEFINE_string(right_rect_topic, "/right/image_rect", "point_cloud topic name");

DEFINE_string(calib_file, "calibration.yaml", "Stereo calibration file name");
DEFINE_string(input_bag,"example.bag", "Input bag");
DEFINE_int32(calib_width, 640, "Calibration image width");
DEFINE_int32(calib_height, 480, "Calibration image height");

DEFINE_bool(debug, false, "Debug Mode to visualize rectified imgs");
DEFINE_int32(algo, 1, "Which stereo alhorithm to run ELAS =1 , SGBM=2");
DEFINE_bool(calibrated, true, "specify calibrated pair or not");
DEFINE_bool(write_src_imgs, false, "specify to write raw imgs to bag");
DEFINE_bool(write_rect_imgs, false, "specify to write rect imgs to bag");
DEFINE_bool(write_disparity_img, true, "specify to write disparity to bag");
DEFINE_bool(publish2ros, false, "specify to publish disparity and rect to ros");
DEFINE_bool(visualize, false, "specify to visualize imgs // manual steps");
DEFINE_bool(disparity_colormap, false, "True: to apply COLORMAP_JET for disparity");
DEFINE_bool(approx_policy, false, "specify to use approximate_time policy");

Size out_img_size;
Size calib_img_size;
FileStorage calib_file;
rosbag::Bag outBag_;


vector<Mat> K_mats , D_mats, R_mats, T_mats, P_mats;
//image_transport::CameraPublisher lRectImgPub_, rRectImgPub_;
image_transport::Publisher dispMapPub_, lRectImgPub_, rRectImgPub_;
//ros::Publisher pcl_pub;

DepthReconstructor* depthProc;
depthUncalibrated* uncalibDepthProc;

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public: 
  void newMessage(const boost::shared_ptr<M const>& msg)
  {
    this->signalMessage(msg);
  }
};

void processCalibratedPair(Mat& leftImg, Mat& rightImg, Mat& rectLeft, Mat& rectRight, Mat& dispImg)
{
    Mat depthMap;
    depthProc->calcDisparity(leftImg, rightImg, dispImg, depthMap);
    rectLeft = depthProc->img1_rect;
    rectRight = depthProc->img2_rect;
}

void processUncalibratedPair(Mat& leftImg, Mat& rightImg, Mat& rectLeft, Mat& rectRight, Mat& dispImg)
{
    uncalibDepthProc->calcDisparity(leftImg, rightImg, dispImg, rectLeft,rectRight);
}

void imgCallback(const sensor_msgs::ImageConstPtr& msg_left, 
                const sensor_msgs::ImageConstPtr& msg_right) 
{
    Mat tmpL = cv_bridge::toCvShare(msg_left, "mono8")->image;
    Mat tmpR = cv_bridge::toCvShare(msg_right, "mono8")->image;
    if (tmpL.empty() || tmpR.empty())
        return;
    
    Mat dispImg, rectLeft, rectRight;

    if(FLAGS_calibrated)
    {
        processCalibratedPair(tmpL, tmpR, rectLeft, rectRight, dispImg);
    }
    else
    {
        processUncalibratedPair(tmpL, tmpR, rectLeft, rectRight, dispImg);
    }

    if (dispImg.empty())
        return;

    sensor_msgs::ImagePtr msgDisparity;
    cv::Mat dispColor, disp8bit, disparityNormalized;
    if(FLAGS_disparity_colormap)
    {
        cv::cvtColor(dispImg, dispColor, cv::COLOR_GRAY2BGR);
        dispColor.convertTo(disp8bit, CV_8UC3, 1);
        cv::normalize(disp8bit, disparityNormalized, 0, 256, cv::NORM_MINMAX, CV_8UC3);
        cv::applyColorMap(disparityNormalized,disparityNormalized,cv::COLORMAP_JET);

        // write to bag
        msgDisparity = cv_bridge::CvImage(msg_left->header, "bgr8", disparityNormalized).toImageMsg();
    }
    else
    {
        dispImg.convertTo(disp8bit, CV_8UC1, 1);
        cv::normalize(disp8bit, disparityNormalized, 0, 256, cv::NORM_MINMAX, CV_8UC1);

        // write to bag
        msgDisparity = cv_bridge::CvImage(msg_left->header, "mono8", disparityNormalized).toImageMsg();
    }

    if(FLAGS_write_src_imgs)
    {
        outBag_.write(FLAGS_left_topic,  msg_left->header.stamp, msg_left);
        outBag_.write(FLAGS_right_topic, msg_right->header.stamp, msg_right);
    }
    
    sensor_msgs::ImagePtr rectLeftMsg, rectRightMsg;
    rectLeftMsg = cv_bridge::CvImage(msg_left->header, "mono8", rectLeft).toImageMsg();
    rectRightMsg = cv_bridge::CvImage(msg_right->header, "mono8", rectRight).toImageMsg();
/*    sensor_msgs::CameraInfoPtr leftRectInfoMsg, rightRectInfoMsg;
    leftRectInfoMsg->header = msg_left->header;
    rightRectInfoMsg->header = msg_right->header;*/

    if(FLAGS_write_rect_imgs)
    {
        outBag_.write(FLAGS_left_rect_topic,  msg_left->header.stamp, rectLeftMsg);
        outBag_.write(FLAGS_right_rect_topic, msg_right->header.stamp, rectRightMsg);
    }

    if(FLAGS_write_disparity_img)
        outBag_.write(FLAGS_disparity_topic, msg_left->header.stamp, msgDisparity);
    if(FLAGS_publish2ros)
    {
        dispMapPub_.publish(msgDisparity);
        //lRectImgPub_.publish(rectLeftMsg,leftRectInfoMsg);
        //rRectImgPub_.publish(rectRightMsg,rightRectInfoMsg);
        lRectImgPub_.publish(rectLeftMsg);
        rRectImgPub_.publish(rectRightMsg);
    }
    // viz rect pair
    if(FLAGS_debug)
    {
        Mat pair;
        pair.create(calib_img_size.height, calib_img_size.width * 2, CV_8U);
        rectLeft.copyTo(pair.colRange(0, calib_img_size.width));
        rectRight.copyTo(pair.colRange(calib_img_size.width, calib_img_size.width * 2));
        for (int j = 0; j < calib_img_size.height; j += 50)
            cv::line(pair, cv::Point(0, j), cv::Point(calib_img_size.width * 2, j), cv::Scalar(255));
        cv::namedWindow("rectified",0);
        cv::imshow("rectified", pair);
        cv::resizeWindow("rectified", 640*2, 480);

        cv::waitKey(0);
    }

    if (FLAGS_visualize)
    {
        std::cout<<" disparity "<<dispImg.rows<<" "<<dispImg.cols<<std::endl;
        cv::imshow("disparity image", disparityNormalized);
        //cv::waitKey(30);
        int key = cvWaitKey(30);

        if( (key & 255)==27 ) 
        {   // ESC key
            cv::destroyAllWindows();    
            ros::shutdown();
        }
        /* else if( (key & 255)==32 ) 
        {   //space key
            return;
        }*/
    }
}

void loadBag(const rosbag::Bag& bag)
{

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back(FLAGS_left_topic);
  topics.push_back(FLAGS_right_topic);

  rosbag::View view(bag, rosbag::TopicQuery(topics));
  // Set up fake subscribers to capture images
  BagSubscriber<sensor_msgs::Image> l_img_sub, r_img_sub;
  
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync_exact(l_img_sub, r_img_sub, 25);
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync_approx(MySyncPolicy(25), l_img_sub, r_img_sub);

  if (!FLAGS_approx_policy)
  {
    std::cout << "using exact time policy"<<std::endl;

    sync_exact.registerCallback(boost::bind(&imgCallback, _1, _2));
  }
  else
  {
    std::cout<<"using approx time policy"<<std::endl;
    
    sync_approx.registerCallback(boost::bind(&imgCallback, _1, _2));
  }
  
  // Load all messages into our stereo dataset

  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
    if (!ros::ok())
        break;

    if (m.getTopic() == FLAGS_left_topic || ("/" + m.getTopic() == FLAGS_left_topic))
    {
        sensor_msgs::Image::ConstPtr l_img =m.instantiate<sensor_msgs::Image>();
        if (l_img != NULL)
            l_img_sub.newMessage(l_img);
    }
    
    if (m.getTopic() == FLAGS_right_topic || ("/" + m.getTopic() == FLAGS_right_topic))
    {
        sensor_msgs::Image::ConstPtr r_img =m.instantiate<sensor_msgs::Image>();
        if (r_img != NULL)
            r_img_sub.newMessage(r_img);
    }
  }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "stereo_offline", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    // parse arguments
    google::ParseCommandLineFlags(&argc, &argv, true);

    calib_img_size = Size(FLAGS_calib_width, FLAGS_calib_height);

    calib_file = FileStorage(FLAGS_calib_file, FileStorage::READ);
    cout<<FLAGS_calib_file<<endl;

    std::string in_bag_file = FLAGS_input_bag;
    cout<<FLAGS_input_bag<<endl;
    
    std::size_t found = FLAGS_input_bag.find(".bag");
    std::string outBagName = FLAGS_input_bag.substr(0,found)+"_filtered.bag";
    
    // read all the matrices K, D, R, T
    read_calib_params(calib_file,K_mats,D_mats,R_mats,T_mats,P_mats);
    
    //create stereo object
    if(FLAGS_calibrated)
    {
        depthProc = new DepthReconstructor(FLAGS_algo,  calib_img_size, true, FLAGS_debug);
        depthProc->init(K_mats[0] , D_mats[0], K_mats[1], D_mats[1],R_mats[1], T_mats[1]);
    }
    else
    {
        uncalibDepthProc = new depthUncalibrated(FLAGS_algo, calib_img_size);
        uncalibDepthProc->init(K_mats[0] , D_mats[0], K_mats[1], D_mats[1]);
    }

    if(FLAGS_publish2ros)
    {
        /*lRectImgPub_ = it.advertiseCamera(FLAGS_left_rect_topic, 1);
        rRectImgPub_ = it.advertiseCamera(FLAGS_right_rect_topic, 1);*/
        dispMapPub_ = it.advertise(FLAGS_disparity_topic, 1);
        lRectImgPub_ = it.advertise(FLAGS_left_rect_topic, 1);
        rRectImgPub_ = it.advertise(FLAGS_right_rect_topic, 1);
    }

    // load in bag
    rosbag::Bag inBag;
    inBag.open(in_bag_file, rosbag::bagmode::Read);
    outBag_.open(outBagName, rosbag::bagmode::Write);
    loadBag(inBag);
    // close bags
    inBag.close();
    outBag_.close();
    cv::destroyAllWindows();
    std::cout<<"Done"<<std::endl;
    return 0;
}