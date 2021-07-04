#include "depth_uncalibrated.h"

depthUncalibrated::depthUncalibrated(int algo, Size im_size)
{
	depthAlgo_ = (Algos)algo;
	calib_img_size_ = im_size;
}

depthUncalibrated::~depthUncalibrated()
{
}

void depthUncalibrated::calcDisparity(Mat &img1 , Mat &img2, Mat &disp, Mat& img1Rect, Mat& img2Rect)
{
	try
    {
        Size outImgSize = img1.size();
        cv::Mat lmapx, lmapy, rmapx, rmapy;
        findUncalibratedRectificationMap(img1,img2, outImgSize, lmapx, lmapy, rmapx,  rmapy);
        remap(img1, img1Rect, lmapx, lmapy, cv::INTER_LINEAR);
        std::cout<<"recmap 1"<<std::endl;
        remap(img2, img2Rect, rmapx, rmapy, cv::INTER_LINEAR);
        std::cout<<"recmap 2"<<std::endl;
        disp = generateDisparityMap(img1Rect, img2Rect);
        std::cout<<"gen"<<std::endl;
    }
    catch(...)
    {
        std::cout<<" error"<<std::endl;
    }
}
void depthUncalibrated::init( Mat K1, Mat D1, Mat K2, Mat D2 )
{
	K1_ = K1;
    D1_ = D1;
    K2_ = K2;
    D2_ = D2;
}
void depthUncalibrated::findUncalibratedRectificationMap(Mat imgLeft, Mat imgRight, Size finalSize, cv::Mat& lmapx,cv::Mat& lmapy,cv::Mat& rmapx,cv::Mat& rmapy)
{
    std::cout<<"img size "<<finalSize<<std::endl;
    //get keypoints
    //Ptr<Feature2D> feature =  AKAZE::create();
    Ptr<Feature2D> feature =  ORB::create(2000);
    
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
    std::vector<DMatch> filteredMatches;
    vector<Point2f> src;
    vector<Point2f> dst;
    int num_neighbors = 2;
    double dist_thresh = 0.6;
    matcher->knnMatch(desc1, desc2, matches, num_neighbors);
    //Go through the matches -> gives m = vector<DMatch>
    for (auto &m : matches) 
    {
        //check for lowe's ratio threshold and store the match only if the matches are unique.
        // matched point2f of keypoints are stored in src and dst vectors , their indices are stored in i_kp and j_kp
        //if there is only one match then store it
        if (m.size() == 1 or m[0].distance < dist_thresh * m[1].distance) 
        {
            src.push_back(kp1[m[0].queryIdx].pt);
            dst.push_back(kp2[m[0].trainIdx].pt);
            filteredMatches.push_back(m[0]);
        }
    }
    std::cout << " num of matched point "<< src.size()<<std::endl;

    //std::vector<char> outMask(filteredMatches.size());
    cv::Mat outMask;
    //calculate fundamental matrix
    cv::Mat F = findFundamentalMat(cv::Mat(src), cv::Mat(dst), cv::FM_RANSAC, 1, 0.95, outMask);


    cv::Mat outDebug;
    cv::drawMatches(imgLeft, kp1, imgRight, kp2, filteredMatches,outDebug, cv::Scalar::all(-1), cv::Scalar::all(-1), outMask, 0);
    cv::namedWindow("matches",0);
    cv::imshow("matches", outDebug);
    cv::resizeWindow("matches", 640*2, 480);
    cv::waitKey(30);
    vector<Point2f> src1;
    vector<Point2f> dst1;
    std::cout<<outMask<<std::endl;
    std::cout<<outMask.rows << " "<<outMask.cols<<std::endl;
    int count = 0;
    for (int i=0; i<outMask.rows;i++)
    {
        //std::cout<<i<<" "<<int(outMask.at<uchar>(i))<<std::endl;
        //" "<<outMask.at<uchar>(0,i)<<std::endl;
        if (outMask.at<uchar>(i) == 1)
        {
            //std::cout<<"in loop" << ;
            src1.push_back(src[count]);
            dst1.push_back(dst[count]);
        }
        count++;
    }
    std::cout<<" row counts "<< count <<std::endl;

    //rectify uncalibrated
    std::cout<<"F "<<F.size()<< " "<< F<<std::endl;
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(src1, dst1, F, finalSize, H1, H2, 1);
    std::cout<<"H1 "<<H1.size()<< " "<< H1<<std::endl;
    std::cout<<"H2 "<<H2.size()<< " "<< H2<<std::endl;

    //find rectifictaion matrices
    int ref_cam_ind =0;

    Mat Rect1, Rect2;
    //std::cout<<"k1 "<<K1_.size()<< " "<< K1_<<std::endl;
    //std::cout<<"d1 "<<D1_.size()<< " "<< D1_<<std::endl;
    //std::cout<<"k2 "<<K2_.size()<< " "<< K2_<<std::endl;
    //std::cout<<"D2 "<<D2_.size()<< " "<< D2_<<std::endl;
    Rect1 = K1_.inv() * H1 * K1_;
    Rect2 = K2_.inv() * H2 * K2_;
    std::cout<<"Rect1 "<<Rect1.size()<< " "<< Rect1<<std::endl;
    std::cout<<"Rect2 "<<Rect2.size()<< " "<< Rect2<<std::endl;

    //get rectification maps
    cv::initUndistortRectifyMap(K1_, D1_, Rect1, K1_, finalSize, CV_32F, lmapx, lmapy);
    cv::initUndistortRectifyMap(K2_, D2_, Rect2, K2_, finalSize, CV_32F, rmapx, rmapy);
    cout << "Done rectification Uncalibrated" << endl;
/*    std::cout<<"lmapx "<<lmapx.size()<< " "<< lmapx<<std::endl;
    std::cout<<"lmapy "<<lmapy.size()<< " "<< lmapy<<std::endl;
    std::cout<<"rmapx "<<rmapx.size()<< " "<< rmapx<<std::endl;
    std::cout<<"rmapy "<<rmapy.size()<< " "<< rmapy<<std::endl;*/

}
cv::Mat depthUncalibrated::generateDisparityMap(Mat& left, Mat& right) {
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
  //cv::Size depthImgsize = img1.size();
  //Mat dmap = Mat(depthImgsize, CV_8UC1, Scalar(0));
  //leftdpf.convertTo(dmap, CV_8U, 1.);
  return leftdpf.clone();
}