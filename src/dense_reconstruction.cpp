#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <geometry_msgs/Point32.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <stereo_reconstruction/CamToRobotCalibParamsConfig.h>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "elas.h"
#include "popt_pp.h"
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <matplotlibcpp.h>
#include <cmath>
#include "utils.h"

namespace plt = matplotlibcpp;

using namespace cv;
using namespace std;

Mat XR, XT, Q, R, T;
Mat Prect1, Prect2;
Mat Rect1, Rect2, Rect3, Rect4, Rect5;

Mat K1, K2;
Mat D1, D2;

vector<Mat> K_mats , D_mats, R_mats, T_mats, P_mats;

Mat lmapx, lmapy, rmapx, rmapy;
//Vec3d T;

stereo_reconstruction::CamToRobotCalibParamsConfig config;
FileStorage calib_file;
int debug = 0;
Size out_img_size;
Size calib_img_size;
bool with_q = false;
bool rectify_uncalib= false;
bool done_rectify = false;
bool imsave= false;
bool is_ros = false;
int mouseClickX ;
int mouseClickY ;

bool click = false;
int ref_cam_ind = 0;

image_transport::Publisher dmap_pub;
ros::Publisher pcl_pub;

Mat composeRotationCamToRobot(float x, float y, float z) {
  Mat X = Mat::eye(3, 3, CV_64FC1);
  Mat Y = Mat::eye(3, 3, CV_64FC1);
  Mat Z = Mat::eye(3, 3, CV_64FC1);

  X.at<double>(1,1) = cos(x);
  X.at<double>(1,2) = -sin(x);
  X.at<double>(2,1) = sin(x);
  X.at<double>(2,2) = cos(x);

  Y.at<double>(0,0) = cos(y);
  Y.at<double>(0,2) = sin(y);
  Y.at<double>(2,0) = -sin(y);
  Y.at<double>(2,2) = cos(y);

  Z.at<double>(0,0) = cos(z);
  Z.at<double>(0,1) = -sin(z);
  Z.at<double>(1,0) = sin(z);
  Z.at<double>(1,1) = cos(z);

  return Z*Y*X;
}

Mat composeTranslationCamToRobot(float x, float y, float z) {
  return (Mat_<double>(3,1) << x, y, z);
}

void publishPointCloud(Mat& img_left, Mat& dmap) {
  if (debug == 1) {
    XR = composeRotationCamToRobot(config.PHI_X,config.PHI_Y,config.PHI_Z);
    XT = composeTranslationCamToRobot(config.TRANS_X,config.TRANS_Y,config.TRANS_Z);
    cout << "Rotation matrix: " << XR << endl;
    cout << "Translation matrix: " << XT << endl;
  }
  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);
  vector< Point3d > points;
  sensor_msgs::PointCloud pc;
  sensor_msgs::ChannelFloat32 ch;
  ch.name = "rgb";
  pc.header.frame_id = "jackal";
  pc.header.stamp = ros::Time::now();
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      int d = dmap.at<uchar>(j,i);
      // if low disparity, then ignore
      if (d < 2) {
        continue;
      }
      // V is the vector to be multiplied to Q to get
      // the 3D homogenous coordinates of the image point
      V.at<double>(0,0) = (double)(i);
      V.at<double>(1,0) = (double)(j);
      V.at<double>(2,0) = (double)d;
      V.at<double>(3,0) = 1.;
      double X,Y,Z;
      if(with_q && !rectify_uncalib){
          pos = Q * V; // 3D homogeneous coordinate
          X = pos.at<double>(0,0) / pos.at<double>(3,0);
          Y = pos.at<double>(1,0) / pos.at<double>(3,0);
          Z = pos.at<double>(2,0) / pos.at<double>(3,0);
      }
      else{
          double base = -1.0 *Prect2.at<double>(0,3)/Prect2.at<double>(0,0);
          X= ((double)i-Prect2.at<double>(0,2)) * base/(double)d;
          Y= ((double)j-Prect2.at<double>(1,2)) * base/(double)d;
          Z= Prect2.at<double>(0,0) *base / (double)d;

      }

      Mat point3d_cam = Mat(3, 1, CV_64FC1);
      point3d_cam.at<double>(0,0) = X;
      point3d_cam.at<double>(1,0) = Y;
      point3d_cam.at<double>(2,0) = Z;
      // transform 3D point from camera frame to robot frame
      Mat point3d_robot = XR * point3d_cam + XT;
      points.push_back(Point3d(point3d_robot));
      geometry_msgs::Point32 pt;
      pt.x = point3d_robot.at<double>(0,0);
      pt.y = point3d_robot.at<double>(1,0);
      pt.z = point3d_robot.at<double>(2,0);
      pc.points.push_back(pt);
      int32_t red, blue, green;
      red = img_left.at<Vec3b>(j,i)[2];
      green = img_left.at<Vec3b>(j,i)[1];
      blue = img_left.at<Vec3b>(j,i)[0];
      int32_t rgb = (red << 16 | green << 8 | blue);
      ch.values.push_back(*reinterpret_cast<float*>(&rgb));
    }
  }
  if (!dmap.empty()) {
    sensor_msgs::ImagePtr disp_msg;
    disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
    dmap_pub.publish(disp_msg);
  }
  pc.channels.push_back(ch);
  pcl_pub.publish(pc);
}

Mat generateDisparityMap(Mat& left, Mat& right) {
  if (left.empty() || right.empty())
    return left;
  const Size imsize = left.size();
  const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
  Mat leftdpf = Mat::zeros(imsize, CV_32F);
  Mat rightdpf = Mat::zeros(imsize, CV_32F);

  Elas::parameters param(Elas::ROBOTICS);
  param.postprocess_only_left = true;
  Elas elas(param);
  elas.process(left.data, right.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
  Mat dmap = Mat(out_img_size, CV_8UC1, Scalar(0));
  leftdpf.convertTo(dmap, CV_8U, 1.);
  return dmap;
}

void findUncalibratedRectificationMap(Mat imgLeft, Mat imgRight, Size finalSize){

    //get keypoints
    Ptr<Feature2D> feature =  AKAZE::create();
    Ptr<DescriptorMatcher> matcher= DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    feature->detect(imgLeft, kp1);
    //compute features and fill in imagepose
    feature->compute(imgLeft, kp1, desc1);

    feature->detect(imgRight, kp2);
    //compute features and fill in imagepose
    feature->compute(imgRight, kp2, desc2);

    //find matches
    vector<vector<DMatch>> matches;
    vector<Point2f> src;
    vector<Point2f> dst;
    int no_neighbors = 2;
    double dist_thresh = 0.7;
    matcher->knnMatch(desc1, desc2, matches, no_neighbors);
    //Go through the matches -> gives m = vector<DMatch>
    for (auto &m : matches) {
        //check for lowe's ratio threshold and store the match only if the matches are unique.
        // matched point2f of keypoints are stored in src and dst vectors , their indices are stored in i_kp and j_kp
        //if there is only one match then store it
        if (m.size() == 1 or m[0].distance < dist_thresh * m[1].distance) {
            src.push_back(kp1[m[0].queryIdx].pt);
            dst.push_back(kp2[m[0].trainIdx].pt);
        }

    }

    //calculate fundamental matrix
    cv::Mat F = findFundamentalMat(cv::Mat(src), cv::Mat(dst), cv::FM_8POINT);
    //rectifu uncalibrated
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(src, dst, F, finalSize, H1, H2, 3);
    //find rectifictaion matrices
    Rect1 = K_mats[ref_cam_ind].inv() * H1 * K1;
    Rect2 = K_mats[ref_cam_ind+1].inv() * H2 * K2;

    //get rectification maps
    cv::initUndistortRectifyMap(K_mats[ref_cam_ind], D_mats[ref_cam_ind], Rect1, K_mats[ref_cam_ind], finalSize, CV_32F, lmapx, lmapy);
    cv::initUndistortRectifyMap(K_mats[ref_cam_ind+1], D_mats[ref_cam_ind+1], Rect2, K_mats[ref_cam_ind+1], finalSize, CV_32F, rmapx, rmapy);
    cout << "Done rectification Uncalibrated" << endl;
    done_rectify=true;

}


void imgCallback(const sensor_msgs::ImageConstPtr& msg_left, const sensor_msgs::ImageConstPtr& msg_right) {
    if(done_rectify){
        Mat tmpL = cv_bridge::toCvShare(msg_left, "mono8")->image;
        Mat tmpR = cv_bridge::toCvShare(msg_right, "mono8")->image;
        if (tmpL.empty() || tmpR.empty())
            return;

        Mat img_left, img_right, img_left_color;
        remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR);
        remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
        ROS_INFO_STREAM(std::to_string(msg_left->header.seq));
        ROS_INFO_STREAM(msg_left->header.stamp.sec <<"  "<<msg_left->header.stamp.nsec);
        ROS_INFO_STREAM(msg_right->header.stamp.sec<<"  "<<msg_right->header.stamp.nsec);

        cv::imwrite("/home/auv/testing_bagselector/cam2/"+std::to_string(msg_left->header.seq)+".jpg", img_left );
        cv::imwrite("/home/auv/testing_bagselector/cam3/"+std::to_string(msg_left->header.seq)+".jpg", img_right );
        //img_left = tmpL.clone();
        //img_right = tmpR.clone();
        cvtColor(img_left, img_left_color, CV_GRAY2BGR);

        Mat dmap = generateDisparityMap(img_left, img_right);

        //imshow("LEFT", img_left);
        //imshow("RIGHT", img_right);
        imshow("DISP", dmap);
        Mat pair;
        pair.create(out_img_size.height, out_img_size.width * 2, CV_8U);
        img_left.copyTo(pair.colRange(0, out_img_size.width));
        img_right.copyTo(pair.colRange(out_img_size.width, out_img_size.width * 2));
        for (int j = 0; j < out_img_size.height; j += 50)
            cv::line(pair, cv::Point(0, j), cv::Point(out_img_size.width * 2, j), cv::Scalar(255));
        cv::imshow("rectified", pair);

        publishPointCloud(img_left_color, dmap);

        waitKey(30);
    }
    else{
        Mat tmpL = cv_bridge::toCvShare(msg_left, "mono8")->image;
        Mat tmpR = cv_bridge::toCvShare(msg_right, "mono8")->image;

        findUncalibratedRectificationMap(tmpL, tmpR , out_img_size);
    }

}

void imgCallbackFive(const sensor_msgs::ImageConstPtr& msg_1, const sensor_msgs::ImageConstPtr& msg_2,\
                     const sensor_msgs::ImageConstPtr& msg_3, const sensor_msgs::ImageConstPtr& msg_4){ //, const sensor_msgs::ImageConstPtr& msg_5) {
    //ROS_INFO_STREAM("HERE");
    vector<Mat> imgs ;
    imgs.push_back(cv_bridge::toCvShare(msg_1, "mono8")->image);
    imgs.push_back(Mat::zeros(100,100,CV_8U));
    imgs.push_back(cv_bridge::toCvShare(msg_2, "mono8")->image);
    imgs.push_back(cv_bridge::toCvShare(msg_3, "mono8")->image);
    imgs.push_back(cv_bridge::toCvShare(msg_4, "mono8")->image);

    //if (imgs[0].empty() || imgs[1].empty() || imgs[2].empty() || imgs[3].empty() ||imgs[4].empty() )
    //    return;
    string im_name = std::to_string(msg_1->header.seq);
    for (int i = 0; i<5 ; i++){
        cv::imwrite("/home/auv/testing_bagselector/cam"+std::to_string(i)+"/"+im_name+".jpg", imgs[i] );
    }
    cout<<msg_1->header.stamp.sec <<"  "<<msg_1->header.stamp.nsec;
    cout<<msg_2->header.stamp.sec<<"  "<<msg_2->header.stamp.nsec;
    cout<<msg_3->header.stamp.sec<<"  "<<msg_3->header.stamp.nsec;
    cout<<msg_4->header.stamp.sec<<"  "<<msg_4->header.stamp.nsec;
    //cout<<msg_5->header.stamp.sec<<"  "<<msg_5->header.stamp.nsec;
    /*Mat pair;
        pair.create(out_img_size.height, out_img_size.width * 2, CV_8U);
        img_left.copyTo(pair.colRange(0, out_img_size.width));
        img_right.copyTo(pair.colRange(out_img_size.width, out_img_size.width * 2));
        for (int j = 0; j < out_img_size.height; j += 50)
            cv::line(pair, cv::Point(0, j), cv::Point(out_img_size.width * 2, j), cv::Scalar(255));
        cv::imshow("rectified", pair);

        waitKey(30); */

}


void read_calib_params(FileStorage& calib_file) {

    Mat k_temp, d_temp, r_temp, t_temp, p_temp;

    calib_file["K1"] >> k_temp;
    K_mats.push_back(k_temp.clone());
    calib_file["K2"] >> k_temp;
    K_mats.push_back(k_temp.clone());

    calib_file["K3"] >> k_temp;
    K_mats.push_back(k_temp.clone());

    calib_file["K4"] >> k_temp;
    K_mats.push_back(k_temp.clone());

    calib_file["K5"] >> k_temp;
    K_mats.push_back(k_temp.clone());



    calib_file["D1"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D2"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D3"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D4"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D5"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["XR"] >> XR;
    calib_file["XT"] >> XT;


    //build all extrinsic matrices
    Mat_<double> R_temp = Mat_<double>::zeros(3,3);
    Mat_<double> t_tempp = Mat_<double>::zeros(3,1);

    R_temp(0,0) = 1.0; R_temp(1,1) = 1.0; R_temp(2,2) = 1.0;
    t_tempp(0,0) = 0.0; t_tempp(1,0) = 0.0; t_tempp(2,0) = 0.0;
    Mat Rt = build_Rt(R_temp, t_tempp);
    p_temp = K_mats[0]*Rt;
    R_mats.push_back(R_temp.clone());
    T_mats.push_back(t_tempp.clone());
    P_mats.push_back(p_temp.clone());

    //creating the chain
    calib_file["R2"] >> r_temp;
    calib_file["T2"] >> t_temp;

    Rt = build_Rt(r_temp.clone(), t_temp.clone());
    p_temp = K_mats[1]*Rt;
    R_mats.push_back(r_temp.clone());
    T_mats.push_back(t_temp.clone());
    P_mats.push_back(p_temp.clone());


    calib_file["R3"] >> r_temp;
    calib_file["T3"] >> t_temp;
    r_temp = r_temp.clone()*R_mats[1].clone();
    t_temp = r_temp.clone()*T_mats[1].clone() + t_temp.clone();
    Rt = build_Rt(r_temp, t_temp);
    p_temp = K_mats[2]*Rt;
    R_mats.push_back(r_temp.clone());
    T_mats.push_back(t_temp.clone());
    P_mats.push_back(p_temp.clone());

    calib_file["R4"] >> r_temp;
    calib_file["T4"] >> t_temp;
    r_temp = r_temp.clone()*R_mats[2].clone();
    t_temp = r_temp.clone()*T_mats[2].clone() + t_temp.clone();
    Rt = build_Rt(r_temp, t_temp);
    p_temp = K_mats[3]*Rt;
    R_mats.push_back(r_temp.clone());
    T_mats.push_back(t_temp.clone());
    P_mats.push_back(p_temp.clone());

    calib_file["R5"] >> r_temp;
    calib_file["T5"] >> t_temp;
    r_temp = r_temp.clone()*R_mats[3].clone();
    t_temp = r_temp.clone()*T_mats[3].clone() + t_temp.clone();
    Rt = build_Rt(r_temp, t_temp);
    p_temp = K_mats[4]*Rt;
    R_mats.push_back(r_temp.clone());
    T_mats.push_back(t_temp.clone());
    P_mats.push_back(p_temp.clone());


    // calculate disparity between cams 1 and 3
    R = R_mats[ref_cam_ind + 1];
    T = T_mats[ref_cam_ind + 1];

}



void findRectificationMap(FileStorage& calib_file, Size finalSize) {
  Rect validRoi[2];
  cout << "Starting rectification" << endl;
  //rectification is done for first two cameras
  stereoRectify(K_mats[ref_cam_ind], D_mats[ref_cam_ind], K_mats[ref_cam_ind+1], D_mats[ref_cam_ind+1], calib_img_size, R, T, Rect1, Rect2, Prect1, Prect2, Q,
                CV_CALIB_ZERO_DISPARITY, 0.5, finalSize, &validRoi[0], &validRoi[1]); //0.654848 alpha 1.0 = blackareas 0.0 = zoomedin
  cout<<"\nP rectified 1:";
  cout<<Prect1;
  cout<<"\nP rectified 2:";
  cout<<Prect2;
  cout<<"\nRect1:";
  cout<<Rect1;
  cout<<"\nRect2:";
  cout<<Rect2;
  cout<<"\nvalidROI1:";
  cout<<validRoi[0];

  cv::initUndistortRectifyMap(K_mats[ref_cam_ind], D_mats[ref_cam_ind], Rect1, Prect1, finalSize, CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K_mats[ref_cam_ind+1], D_mats[ref_cam_ind+1], Rect2, Prect2, finalSize, CV_32F, rmapx, rmapy);
  cout << "Done rectification" << endl;
  done_rectify=true;
}

void paramsCallback(stereo_reconstruction::CamToRobotCalibParamsConfig &conf, uint32_t level) {
  config = conf;
}

void get_filenames(string imageList ,string fname, vector<string> &imageNames){
    vector<string> cams;
    /*for (int j =0; j<5 ; j++){
        std::stringstream ss;
        ss << imageList<<"/cam" << j;
        cams.push_back(ss.str());
    }*/
    const char *fpath = (imageList+"/"+fname).c_str();
    FILE *f = fopen((imageList+"/"+fname).c_str(), "rt");
    if (!f) {
        cout << "Cannot open file " << fpath << endl;
        return;
    }
    int i = 0;
    for (;;) {
        char buf[1024];
        string pathe;
        if (!fgets(buf, sizeof(buf) - 3, f))
            break;
        size_t len = strlen(buf);
        while (len > 0 && isspace(buf[len - 1]))
            buf[--len] = '\0';
        if (buf[0] == '#')
            continue;

        imageNames.push_back(string(buf));
        i++;
    }
    fclose(f);

}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        mouseClickX = x;
        mouseClickY = y;
        click = true;
    }
    /*else if  ( event == EVENT_RBUTTONDOWN )
    {
        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if  ( event == EVENT_MBUTTONDOWN )
    {
        cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if ( event == EVENT_MOUSEMOVE )
    {
        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

    } */
}

void plotting_func(vector<double> alphas, vector<double> variance,vector<double> sum_sq_diff,vector<double> means ){

    // Set the size of output image to 1200x780 pixels
    //plt::figure_size(640, 480);
    plt::subplot(1,2,1);
    // Plot a red dashed line from given x and y data.
    plt::plot(alphas, variance,"r-");

    // Set x-axis to interval [0,1000000]
    plt::xlim((*std::min_element(alphas.begin(), alphas.end())), (*std::max_element(alphas.begin(), alphas.end())) );
    plt::ylim(0, 200);
    plt::xlabel("Alpha Z in meters");
    plt::ylabel("Variance of intensities");
    plt::grid(true);

    plt::subplot(1,2,2);
    // Plot a line whose name will show up as "log(x)" in the legend.
    plt::named_plot("sum_sq_diff", alphas, sum_sq_diff);
    plt::xlim((*std::min_element(alphas.begin(), alphas.end())), (*std::max_element(alphas.begin(), alphas.end())) );
    plt::xlabel("Alpha Z in meters");
    plt::ylabel("Sum of squares of differences");
    // Add graph title
    plt::title("Sample figure");
    plt::grid(true);
    // Enable legend.
    plt::legend();
    // Save the image (file format is determined by the extension)
    plt::show();
    return;
}

int main(int argc , char** argv){
    int calib_width, calib_height, out_width, out_height;
    const char* calib_file_name;
    const char* left_img_topic;
    const char* right_img_topic;

    static struct poptOption options[] = {
            { "left_topic",'l',POPT_ARG_STRING,&left_img_topic,0,"Left image topic name","STR" },
            { "right_topic",'r',POPT_ARG_STRING,&right_img_topic,0,"Right image topic name","STR" },
            { "calib_file",'c',POPT_ARG_STRING,&calib_file_name,0,"Stereo calibration file name","STR" },
            { "calib_width",'w',POPT_ARG_INT,&calib_width,0,"Calibration image width","NUM" },
            { "calib_height",'h',POPT_ARG_INT,&calib_height,0,"Calibration image height","NUM" },
            { "out_width",'u',POPT_ARG_INT,&out_width,0,"Rectified image width","NUM" },
            { "out_height",'v',POPT_ARG_INT,&out_height,0,"Rectified image height","NUM" },
            { "debug",'d',POPT_ARG_INT,&debug,0,"Set d=1 for cam to robot frame calibration","NUM" },
            POPT_AUTOHELP
            { NULL, 0, 0, NULL, 0, NULL, NULL }
    };

    POpt popt(NULL, argc, argv, options, 0);
    int c;
    while((c = popt.getNextOpt()) >= 0) {}

    calib_img_size = Size(calib_width, calib_height);
    out_img_size = Size(out_width, out_height);

    calib_file = FileStorage(calib_file_name, FileStorage::READ);

    // read all the matrices K, D, R, T
    read_calib_params(calib_file);
    /*calib_file["K1"] >> K1;
    calib_file["K2"] >> K2;
    calib_file["D1"] >> D1;
    calib_file["D2"] >> D2;
    calib_file["R"] >> R;
    calib_file["T"] >> T;
    calib_file["XR"] >> XR;
    calib_file["XT"] >> XT; */

    if(!rectify_uncalib)
        findRectificationMap(calib_file, out_img_size);

    if (is_ros){
        ros::init(argc, argv, "jpp_dense_reconstruction");
        ros::NodeHandle nh;
        image_transport::ImageTransport it(nh);

        if(imsave){


            message_filters::Subscriber<sensor_msgs::Image> sub_img_1(nh, "/camera_array/cam0/image_raw", 1);
            //message_filters::Subscriber<sensor_msgs::Image> sub_img_2(nh, "/camera_array/cam1/image_raw", 1);
            message_filters::Subscriber<sensor_msgs::Image> sub_img_3(nh, "/camera_array/cam2/image_raw", 1);
            message_filters::Subscriber<sensor_msgs::Image> sub_img_4(nh,  "/camera_array/cam3/image_raw", 1);
            message_filters::Subscriber<sensor_msgs::Image> sub_img_5(nh, "/camera_array/cam4/image_raw", 1);

            typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
            message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_1, /*sub_img_2 ,*/ sub_img_3, sub_img_4, sub_img_5);
            sync.registerCallback(boost::bind(&imgCallbackFive, _1, _2 , _3, _4)); //, _5));

            ros::spin();

        }
        else{

            message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, "/camera_array/cam"+std::to_string(ref_cam_ind)+"/image_raw", 1);
            message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, "/camera_array/cam"+std::to_string(ref_cam_ind + 1)+"/image_raw", 1);

            typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
            message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_left, sub_img_right);
            sync.registerCallback(boost::bind(&imgCallback, _1, _2));

            dynamic_reconfigure::Server<stereo_reconstruction::CamToRobotCalibParamsConfig> server;
            dynamic_reconfigure::Server<stereo_reconstruction::CamToRobotCalibParamsConfig>::CallbackType f;

            f = boost::bind(&paramsCallback, _1, _2);
            server.setCallback(f);

            dmap_pub = it.advertise("/camera/left/disparity_map", 1);
            pcl_pub = nh.advertise<sensor_msgs::PointCloud>("/camera/left/point_cloud",1);

            ros::spin();
        }


    }
    else{


        vector<string> imageNames;
        get_filenames("/home/auv/single_image", "imagenames.txt" , imageNames);
        //Create a window
        namedWindow("My Window", 1);
        //set the callback function for any mouse event
        setMouseCallback("My Window", CallBackFunc, NULL);

        for (int i =0; i< imageNames.size() ; i ++){

            vector<Mat> cam_images;
            cv::Mat img_left = cv::imread("/home/auv/single_image/cam"+std::to_string(ref_cam_ind)+"/"+imageNames[i], 0);
            cam_images.push_back(img_left.clone());

            cv::Mat img_right = cv::imread("/home/auv/single_image/cam"+std::to_string(ref_cam_ind + 1)+"/"+imageNames[i], 0);
            cam_images.push_back(img_right.clone());
            cout<<"intensity "<<(int)cam_images[1].at<uchar>(280,500);

            cv::Mat img_temp = cv::imread("/home/auv/single_image/cam"+std::to_string(ref_cam_ind + 2)+"/"+imageNames[i], 0);
            cam_images.push_back(img_temp.clone());

            img_temp = cv::imread("/home/auv/single_image/cam"+std::to_string(ref_cam_ind + 3)+"/"+imageNames[i], 0);
            cam_images.push_back(img_temp.clone());
            img_temp = cv::imread("/home/auv/single_image/cam"+std::to_string(ref_cam_ind + 4)+"/"+imageNames[i], 0);
            cam_images.push_back(img_temp.clone());

            Mat imgL, imgR, img_left_color;
            remap(img_left, imgL, lmapx, lmapy, cv::INTER_LINEAR);
            remap(img_right, imgR, rmapx, rmapy, cv::INTER_LINEAR);

            Mat dmap = generateDisparityMap(imgL, imgR);

            imshow("My Window", imgL);
            //imshow("RIGHT", img_right);
            imshow("DISP", dmap);
            /*Mat pair;
            pair.create(out_img_size.height, out_img_size.width * 2, CV_8U);
            imgL.copyTo(pair.colRange(0, out_img_size.width));
            imgR.copyTo(pair.colRange(out_img_size.width, out_img_size.width * 2));
            for (int j = 0; j < out_img_size.height; j += 50)
                cv::line(pair, cv::Point(0, j), cv::Point(out_img_size.width * 2, j), cv::Scalar(255));
            cv::imshow("rectified", pair); */

            cvtColor(imgL, img_left_color, CV_GRAY2BGR);

            Mat V = Mat(4, 1, CV_64FC1);
            Mat pos = Mat(4, 1, CV_64FC1);
            vector< Point3d > points;

            //waitKey(0);
            int key = waitKey(0) & 0xFF ;

            while ( key != (int)'a' or click){
                if (click){
                    Mat mouseClick_World = Mat(3,1, CV_64FC1);

                    int d = dmap.at<uchar>(mouseClickY,mouseClickX);
                    // if low disparity, then ignore
                    if (d < 2) {
                        continue;
                        click=false;
                        key = waitKey(0) & 0xFF ;
                    }

                    //calculate the 3D point for the disparity
                    double X,Y,Z;
                    double base = -1.0 *Prect2.at<double>(0,3)/Prect2.at<double>(0,0);
                    X= ((double)mouseClickX-Prect2.at<double>(0,2)) * base/(double)d;
                    Y= ((double)mouseClickY-Prect2.at<double>(1,2)) * base/(double)d;
                    Z= Prect2.at<double>(0,0) *base / (double)d;
                    cout <<"\n X Y Z in rectified frame : ";
                    cout<<X<<","<<Y<<","<<Z<<"\n";

                    //get the 3D in original frame. convert from rectified frame.
                    Mat iRectL = Rect1.inv();
                    Mat w_in_rect = Mat(3, 1, CV_64FC1);
                    w_in_rect.at<double>(0,0) = X;
                    w_in_rect.at<double>(1,0) = Y;
                    w_in_rect.at<double>(2,0) = Z;
                    Mat w_in_orig = iRectL * w_in_rect;
                    cout <<"\n X Y Z in original frame : ";
                    cout<< w_in_orig.at<double>(0,0)<<","<< w_in_orig.at<double>(1,0)<<","<<w_in_orig.at<double>(2,0)<<"\n";


                    //get the corresponding non-rectified coordinates. This chunk of code is only for debug. We wanna
                    // reconstruct the 3D point using unrectified image and check if it matches with tranformed 3D point in original frame (w_in_orig)
                    int unrect_x = lmapx.at<float>(mouseClickY,mouseClickX);
                    int unrect_y = lmapy.at<float>(mouseClickY,mouseClickX);
                     cout<<unrect_x<<","<<unrect_y<<"\n";
                    double n_X = ((double)unrect_x - K_mats[ref_cam_ind].at<double>(0,2))/K_mats[ref_cam_ind].at<double>(0,0) * w_in_orig.at<double>(2,0);
                    double n_Y = ((double)unrect_y - K_mats[ref_cam_ind].at<double>(1,2))/K_mats[ref_cam_ind].at<double>(1,1) * w_in_orig.at<double>(2,0);
                    double n_Z = w_in_orig.at<double>(2,0);
                    cout<<n_X<<","<<n_Y<<","<<n_Z<<"\n";

                    mouseClick_World.at<double>(0,0) = w_in_orig.at<double>(0,0) / w_in_orig.at<double>(2,0);
                    mouseClick_World.at<double>(1,0) = w_in_orig.at<double>(1,0) / w_in_orig.at<double>(2,0);
                    mouseClick_World.at<double>(2,0) = 1.0;

                    cout<<"\n Reconstructed point : "<<mouseClick_World;
                    cout<<"\n new point   variance    sun_sq_diff   mean \n";
                    double del_Z = 0.01;
                    int n = 50;
                    vector<double> variance, sum_sq_diff, alphas, means;
                    for ( int j =0; j < n ; j ++){
                        double new_z =  w_in_orig.at<double>(2,0) + del_Z * (j - n/2);
                        Mat alpha_Z =  Mat(4,1, CV_64FC1);
                        alpha_Z.at<double>(0,0) = new_z * mouseClick_World.at<double>(0,0);
                        alpha_Z.at<double>(1,0) = new_z * mouseClick_World.at<double>(1,0);
                        alpha_Z.at<double>(2,0) = new_z * mouseClick_World.at<double>(2,0);
                        alpha_Z.at<double>(3,0) = 1.0;

                        //project it to different cameras frames
                        vector<Mat> new_coords;
                        double mean=0.0;
                        vector<double> intensities;
                        double s_sq_diff=0.0;
                        mean = mean + (int)imgL.at<uchar>(mouseClickY,mouseClickX);
                        intensities.push_back((int)imgL.at<uchar>(mouseClickY,mouseClickX));

                        for(int c_ind=1; c_ind<5; c_ind++){
                            Mat temp = P_mats[c_ind] * alpha_Z;
                            Mat temp2 = Mat(2,1, CV_16U);
                            temp2.at<int>(0,0) = ((int) temp.at<double>(0,0)/temp.at<double>(2,0));
                            temp2.at<int>(1,0) = (int) temp.at<double>(1,0)/temp.at<double>(2,0);
                            new_coords.push_back(temp2.clone());

                            //cout<<temp2;

                            //cout<<(int)imgL.at<uchar>(mouseClickY,mouseClickX);

                            int row =temp2.at<uint16_t >(1,0);
                            int col = temp2.at<uint16_t >(0,0);

                            s_sq_diff = s_sq_diff + abs((int)imgL.at<uchar>(mouseClickY,mouseClickX) - (int)cam_images[c_ind].at<uchar>(row, col));
                            mean = mean + (int)cam_images[c_ind].at<uchar>(row, col);
                            intensities.push_back((int)cam_images[c_ind].at<uchar>(row, col));
                            //cout<<"\nIntensity diff : "<<;
                        }

                        //calc variance
                        mean = mean/5;
                        double var=0.0;
                        for ( int c_ind =0; c_ind <5 ; c_ind++){
                            var = var + pow((intensities[c_ind] - mean), 2);

                        }
                        var = var/5;
                        variance.push_back(var);
                        s_sq_diff = s_sq_diff/5;
                        sum_sq_diff.push_back(s_sq_diff);
                        cout<< alpha_Z.at<double>(2,0)<<"    "<<var<<"    "<<s_sq_diff<<"    "<<mean<<" \n";
                        alphas.push_back(alpha_Z.at<double>(2,0));
                        means.push_back(mean);


                       /* Mat temp = Prect2 * alpha_Z;

                        int new_u = (int) temp.at<double>(0,0)/temp.at<double>(2,0);
                        int new_v = (int) temp.at<double>(1,0)/temp.at<double>(2,0); */



                    }
                    // Plot the results
                    plotting_func(alphas, variance, sum_sq_diff, means );
                    click = false;
                }
                key = waitKey(0) & 0xFF ;
            }
        }
    }


}


int main_n(int argc, char** argv) {
    int calib_width, calib_height, out_width, out_height;
    const char* calib_file_name;
    const char* left_img_topic;
    const char* right_img_topic;

  static struct poptOption options[] = {
    { "left_topic",'l',POPT_ARG_STRING,&left_img_topic,0,"Left image topic name","STR" },
    { "right_topic",'r',POPT_ARG_STRING,&right_img_topic,0,"Right image topic name","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file_name,0,"Stereo calibration file name","STR" },
    { "calib_width",'w',POPT_ARG_INT,&calib_width,0,"Calibration image width","NUM" },
    { "calib_height",'h',POPT_ARG_INT,&calib_height,0,"Calibration image height","NUM" },
    { "out_width",'u',POPT_ARG_INT,&out_width,0,"Rectified image width","NUM" },
    { "out_height",'v',POPT_ARG_INT,&out_height,0,"Rectified image height","NUM" },
    { "debug",'d',POPT_ARG_INT,&debug,0,"Set d=1 for cam to robot frame calibration","NUM" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  calib_img_size = Size(calib_width, calib_height);
  out_img_size = Size(out_width, out_height);

  calib_file = FileStorage(calib_file_name, FileStorage::READ);
  calib_file["K1"] >> K1;
  calib_file["K2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  calib_file["R"] >> R;
  calib_file["T"] >> T;
  calib_file["XR"] >> XR;
  calib_file["XT"] >> XT;
  //calib_file["P1"] >> P1;
  //calib_file["P2"] >> P2;

  if(!rectify_uncalib)
      findRectificationMap(calib_file, out_img_size);


    if (is_ros){
        ros::init(argc, argv, "jpp_dense_reconstruction");
        ros::NodeHandle nh;
        image_transport::ImageTransport it(nh);

        message_filters::Subscriber<sensor_msgs::Image> sub_img_left(nh, left_img_topic, 1);
        message_filters::Subscriber<sensor_msgs::Image> sub_img_right(nh, right_img_topic, 1);

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
        message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_img_left, sub_img_right);
        sync.registerCallback(boost::bind(&imgCallback, _1, _2));

        dynamic_reconfigure::Server<stereo_reconstruction::CamToRobotCalibParamsConfig> server;
        dynamic_reconfigure::Server<stereo_reconstruction::CamToRobotCalibParamsConfig>::CallbackType f;

        f = boost::bind(&paramsCallback, _1, _2);
        server.setCallback(f);

        dmap_pub = it.advertise("/camera/left/disparity_map", 1);
        pcl_pub = nh.advertise<sensor_msgs::PointCloud>("/camera/left/point_cloud",1);

        ros::spin();

    }else{

        vector<string> imageNames;
        get_filenames("/home/auv/testing_bagselector", "imagenames.txt" , imageNames);
        //Create a window
        namedWindow("My Window", 1);
        //set the callback function for any mouse event
        setMouseCallback("My Window", CallBackFunc, NULL);

        for (int i =0; i< imageNames.size() ; i ++){
            cv::Mat imgL = cv::imread("/home/auv/testing_bagselector/cam2/"+imageNames[i], 0);
            cv::Mat imgR = cv::imread("/home/auv/testing_bagselector/cam3/"+imageNames[i], 0);

            Mat dmap = generateDisparityMap(imgL, imgR);

            imshow("My Window", imgL);
            //imshow("RIGHT", img_right);
            imshow("DISP", dmap);
            /*Mat pair;
            pair.create(out_img_size.height, out_img_size.width * 2, CV_8U);
            imgL.copyTo(pair.colRange(0, out_img_size.width));
            imgR.copyTo(pair.colRange(out_img_size.width, out_img_size.width * 2));
            for (int j = 0; j < out_img_size.height; j += 50)
                cv::line(pair, cv::Point(0, j), cv::Point(out_img_size.width * 2, j), cv::Scalar(255));
            cv::imshow("rectified", pair); */
            Mat img_left_color;
            cvtColor(imgL, img_left_color, CV_GRAY2BGR);

            Mat V = Mat(4, 1, CV_64FC1);
            Mat pos = Mat(4, 1, CV_64FC1);
            vector< Point3d > points;

            //waitKey(0);
            int key = waitKey(0) & 0xFF ;

            while ( key != (int)'a' or click){
                if (click){
                    Mat mouseClick = Mat(3,1, CV_64FC1);

                    int d = dmap.at<uchar>(mouseClickY,mouseClickX);
                    // if low disparity, then ignore
                    if (d < 2) {
                        continue;
                    }
                    double X,Y,Z;
                    double base = -1.0 *Prect2.at<double>(0,3)/Prect2.at<double>(0,0);
                    X= ((double)mouseClickX-Prect2.at<double>(0,2)) * base/(double)d;
                    Y= ((double)mouseClickY-Prect2.at<double>(1,2)) * base/(double)d;
                    Z= Prect2.at<double>(0,0) *base / (double)d;

                    mouseClick.at<double>(0,0) = X/Z;
                    mouseClick.at<double>(1,0) = Y/Z;
                    mouseClick.at<double>(2,0) = 1.0;

                    cout<<"\n Reconstructed point : "<<mouseClick;
                    double del_Z = 0.1;
                    int n = 10;
                    for ( int j =0; j < n ; j ++){
                        double new_z = Z + del_Z * (j - n/2);
                        Mat alpha_Z =  Mat(4,1, CV_64FC1);
                        alpha_Z.at<double>(0,0) = new_z * mouseClick.at<double>(0,0);
                        alpha_Z.at<double>(1,0) = new_z * mouseClick.at<double>(1,0);
                        alpha_Z.at<double>(2,0) = new_z * mouseClick.at<double>(2,0);
                        alpha_Z.at<double>(3,0) = 1.0;

                        Mat temp = Prect2 * alpha_Z;

                        int new_u = (int) temp.at<double>(0,0)/temp.at<double>(2,0);
                        int new_v = (int) temp.at<double>(1,0)/temp.at<double>(2,0);

                        cout<<"\n new point : "<<alpha_Z;
                        cout<<"\n new coords : ("<<new_u <<","<<new_v<<")";
                        cout<<"\nIntensity diff right Image: "<<abs((int)imgL.at<uchar>(mouseClickY,mouseClickX) - (int)imgR.at<uchar>(new_v, new_u));




                    }

                   click = false;
                }
                key = waitKey(0) & 0xFF ;
            }
        }


    }



    return 0;
}