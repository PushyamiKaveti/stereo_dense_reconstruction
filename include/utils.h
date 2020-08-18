//
// Created by auv on 7/11/20.
//

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat build_Rt(Mat R, Mat t);
void read_calib_params(FileStorage& calib_file, vector<Mat>& K_mats , vector<Mat>& D_mats, vector<Mat>& R_mats, vector<Mat>& T_mats, vector<Mat>& P_mats) ;
string pathJoin( const string& p1, const string& p2);