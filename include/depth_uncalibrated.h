#include <string>
#include <iostream>
#include <sys/stat.h>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <ctime>
#include <cmath>

#include "utils.h"

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "elas.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

enum Algos {
    ELAS1 = 1,
    BLOCK_MATCH1 = 2
};

class depthUncalibrated
{
public:
	depthUncalibrated(int algo, Size im_size);
	~depthUncalibrated();
	void calcDisparity(Mat &img1 , Mat &img2, Mat &disp, Mat& img1Rect, Mat& img2Rect);
	void init( Mat K1, Mat D1, Mat K2, Mat D2 ); 
	
private:
	Algos depthAlgo_; 
	Size calib_img_size_;
	cv::Mat K1_, D1_, K2_, D2_;
	void findUncalibratedRectificationMap(Mat imgLeft, Mat imgRight, Size finalSize, cv::Mat& lmapx,cv::Mat& lmapy,cv::Mat& rmapx,cv::Mat& rmapy);
	cv::Mat generateDisparityMap(Mat& left, Mat& right);
};