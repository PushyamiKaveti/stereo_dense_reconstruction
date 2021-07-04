//
// Created by auv on 7/11/20.
//

#include "utils.h"
using namespace cv;
using namespace std;

Mat build_Rt(Mat R, Mat t) {

    Mat_<double> Rt = Mat_<double>::zeros(3,4);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Rt(i,j) = R.at<double>(i,j);
        }
        Rt(i,3) = t.at<double>(0,i);
    }
    return Rt;

}
void read_calib_params(FileStorage& calib_file, vector<Mat>& K_mats , vector<Mat>& D_mats, vector<Mat>& R_mats, vector<Mat>& T_mats, vector<Mat>& P_mats) {
    Mat k_temp, d_temp, r_temp, t_temp, p_temp;
    K_mats.clear();
    D_mats.clear();
    R_mats.clear();
    T_mats.clear();
    P_mats.clear();
    // K Matrices
    calib_file["K1"] >> k_temp;
    K_mats.push_back(k_temp.clone());
    calib_file["K2"] >> k_temp;
    K_mats.push_back(k_temp.clone());

    /*calib_file["K3"] >> k_temp;
    K_mats.push_back(k_temp.clone());

    calib_file["K4"] >> k_temp;
    K_mats.push_back(k_temp.clone());

    calib_file["K5"] >> k_temp;
    K_mats.push_back(k_temp.clone()); */

    //Distortion vectors
    calib_file["D1"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D2"] >> d_temp;
    D_mats.push_back(d_temp.clone());
    /*calib_file["D3"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D4"] >> d_temp;
    D_mats.push_back(d_temp.clone());

    calib_file["D5"] >> d_temp;
    D_mats.push_back(d_temp.clone());*/

   // calib_file["XR"] >> XR;
   // calib_file["XT"] >> XT;


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
    calib_file["R"] >> r_temp;
    calib_file["T"] >> t_temp;
    Rt = build_Rt(r_temp.clone(), t_temp.clone());
    p_temp = K_mats[1]*Rt;
    R_mats.push_back(r_temp.clone());
    T_mats.push_back(t_temp.clone());
    P_mats.push_back(p_temp.clone());

}

string pathJoin( const string& p1, const string& p2) {

    char sep = '/';
    string tmp = p1;

    if (p1[p1.length()] != sep) { // Need to add a
        tmp += sep;
        tmp +=p2;// path separator

    }
    else{
        tmp += p2;
    }
    return tmp;
}
