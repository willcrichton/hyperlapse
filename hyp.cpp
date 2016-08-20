#include <iostream>
#include <algorithm>
#include <limits>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "CycleTimer.h"

using namespace std;
using namespace cv;

#define START(S)                                \
  LOG(INFO) << S << "...";                      \
  start = CycleTimer::currentSeconds();         \


#define END {                                                           \
    LOG(INFO) << "Done in " << CycleTimer::currentSeconds() - start << "s."; \
  }

class Constants {
public:
  int w = 32;
  int g = 4;
  int iw, ih;
  int T;
  int d;
  float tau_c, gamma;
  int lam_s = 200;
  int lam_a = 80;
  int tau_s = 200;
  int tau_a = 200;
  int v = 8;

  Constants(int iw, int ih, int T) {
    this->iw = iw;
    this->ih = ih;
    this->T = T;
    d = (int) sqrt((float)(ih * ih + iw * iw));
    tau_c = 0.1 * (float) d;
    gamma = 0.5 * (float) d;
  }
};

float reprojection_error(vector<Point2f>& src, vector<Point2f>& dst, Mat& H) {
  vector<Point2f> dst_proj;
  perspectiveTransform(src, dst_proj, H);
  int N = src.size();
  Mat dst_proj_m = Mat::zeros(N, 2, CV_32F), dst_m = Mat::zeros(N, 2, CV_32F);
  for (int i = 0; i < N; i++) {
    dst_proj_m.at<float>(i, 0) = dst_proj[i].x;
    dst_proj_m.at<float>(i, 1) = dst_proj[i].y;
    dst_m.at<float>(i, 0) = dst[i].x;
    dst_m.at<float>(i, 1) = dst[i].y;
  }
  Mat diff = dst_m - dst_proj_m;
  Mat summed, sq;
  reduce(diff.mul(diff), summed, 1, CV_REDUCE_SUM);
  sqrt(summed, sq);
  return mean(sq)[0];
}

float match_cost(Constants C,
                 vector<KeyPoint>& kp1, Mat& desc1,
                 vector<KeyPoint>& kp2, Mat& desc2)
{
  FlannBasedMatcher matcher;
  vector<DMatch> matches;
  matcher.match(desc1, desc2, matches);

  double min_dist = numeric_limits<double>::max();
  for (auto& match : matches) {
    double dist = match.distance;
    if (dist < min_dist) { min_dist = dist; }
  }

  vector<Point2f> fr1, fr2;
  for (auto& match : matches) {
    if (match.distance <= max(2 * min_dist, 0.02)) {
      fr1.push_back(kp1[match.queryIdx].pt);
      fr2.push_back(kp2[match.trainIdx].pt);
    }
  }

  // Need at least 4 points to find a homography
  if (fr1.size() < 4) {
    return C.gamma;
  }

  Mat mask;
  Mat H = findHomography(fr1, fr2, CV_RANSAC, 3, mask);
  float cr = reprojection_error(fr1, fr2, H);

  Point2f x(C.ih/2), y(C.iw/2);
  vector<Point2f> center = {x, y};
  float co = reprojection_error(center, center, H);

  // LOG(INFO) << "cr: " << cr << ", co: " << co << ", C.tau_c: " << C.tau_c;

  if (cr < C.tau_c) {
    return co;
  } else {
    return C.gamma;
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  double start;

  VideoCapture input("test.mp4");
  if (!input.isOpened()) {
    cerr << "Input failed to open";
    exit(0);
  }

  START("Loading frames");
  vector<Mat> frames;
  while(frames.size() < 300) {
    Mat frame;
    if (!input.read(frame)) { break; }
    frames.push_back(frame);
  }
  LOG(INFO) << "Loaded " << frames.size() << " frames.";
  END;

  Constants C(frames[0].cols, frames[0].rows, frames.size());

  auto vel_cost = [&](int i, int j) {
    return std::min(powf((j - i) - C.v, 2), (float) C.tau_s);
  };

  auto acc_cost = [&](int h, int i, int j) {
    return std::min(powf((j - i) - (i - h), 2), (float) C.tau_a);
  };

  vector<Mat> features;
  vector<vector<KeyPoint>> kps;
  START("Detecting features");
  for (int i = 0; i < C.T; i++) {
    Mat feat;
    vector<KeyPoint> kp;
    features.push_back(feat);
    kps.push_back(kp);
  }

#pragma omp parallel
  {
    SurfFeatureDetector detector(400);
    SurfDescriptorExtractor extractor;
#pragma omp for schedule(dynamic)
    for (int i = 0; i < frames.size(); i++) {
      detector.detect(frames[i], kps[i]);
      extractor.compute(frames[i], kps[i], features[i]);
    }
  }
  END;

  // int a = 0;
  // int b = 59;

  // Mat img_object = frames[a],
  //   img_scene = frames[b];

  // vector<KeyPoint> keypoints_object = kps[a],
  //   keypoints_scene = kps[b];

  // Mat descriptors_object = features[a],
  //   descriptors_scene = features[b];

  // match_cost(C, keypoints_object, descriptors_object, keypoints_scene, descriptors_scene);

  // //-- Step 3: Matching descriptor vectors using FLANN matcher
  // FlannBasedMatcher matcher;
  // std::vector< DMatch > matches;
  // matcher.match( descriptors_object, descriptors_scene, matches );

  // double max_dist = 0; double min_dist = 100;

  // //-- Quick calculation of max and min distances between keypoints
  // for( int i = 0; i < descriptors_object.rows; i++ )
  // { double dist = matches[i].distance;
  //   if( dist < min_dist ) min_dist = dist;
  //   if( dist > max_dist ) max_dist = dist;
  // }

  // printf("-- Max dist : %f \n", max_dist );
  // printf("-- Min dist : %f \n", min_dist );

  // //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  // std::vector< DMatch > good_matches;

  // for( int i = 0; i < descriptors_object.rows; i++ )
  // { //if( matches[i].distance < 3*min_dist )
  //    { good_matches.push_back( matches[i]); }
  // }

  // Mat img_matches;
  // drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
  //              good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
  //              vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  // std::vector<Point2f> obj;
  // std::vector<Point2f> scene;

  // for( int i = 0; i < good_matches.size(); i++ )
  // {
  //   //-- Get the keypoints from the good matches
  //   obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
  //   scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  // }

  // Mat H = findHomography( obj, scene, CV_RANSAC );

  // //-- Get the corners from the image_1 ( the object to be "detected" )
  // std::vector<Point2f> obj_corners(4);
  // obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  // obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  // std::vector<Point2f> scene_corners(4);

  // perspectiveTransform( obj_corners, scene_corners, H);

  // //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  // line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  // line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  // line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  // //-- Show detected matches
  // imshow( "Good Matches & Object detection", img_matches );

  // waitKey(0);
  // exit(0);

  Mat Cm = Mat::zeros(C.T+1, C.T+1, CV_32F);
  START("Building cost matrix");
  // "parallel for" shorthand doesn't seem to work for some reason?
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 1; i <= C.T; i++) {
      for (int j = i+1; j <= min(i+C.w, C.T); j++) {
        Cm.at<float>(i, j) =
          match_cost(C, kps[i-1], features[i-1], kps[j-1], features[j-1]);
      }
    }
  }
  END;

  START("Finding path");
  Mat Dv = Mat::zeros(C.T+1, C.T+1, CV_32F);
  Mat Tv = Mat::zeros(C.T+1, C.T+1, CV_32S);

  // Initialization
  for (int i = 1; i <= C.g; i++) {
    for (int j = i+1; j <= i+C.w; j++) {
      Dv.at<float>(i, j) = Cm.at<float>(i, j) + C.lam_s * vel_cost(i, j);
    }
  }

  // First pass: populate Dv
  for (int i = C.g; i <= C.T; i++) {
    for (int j = i+1; j <= min(i+C.w, C.T); j++) {
      float c = Cm.at<float>(i, j) + C.lam_s * vel_cost(i, j);
      // LOG(INFO) << i << "," << j << ": " << Cm.at<float>(i, j) << " -- " << C.lam_s * vel_cost(i, j);
      float minv = numeric_limits<float>::max();
      int mink = 0;
      for (int k = 1; k <= min(i-1,C.w); k++) {
        float v = Dv.at<float>(i-k, i) + C.lam_a * acc_cost(i-k, i, j);
        if (v < minv) {
          minv = v;
          mink = k;
        }
      }
      Dv.at<float>(i, j) = c + minv;
      Tv.at<int>(i, j) = i - mink;
      //LOG(INFO) << i << "," << j << ": " << i-mink << " -- " << c + minv;
    }
  }

  // Second pass: trace back min cost path
  int s = 0;
  int d = 0;
  float dmin = numeric_limits<float>::max();
  for (int i = C.T-C.g; i <= C.T; i++) {
    for (int j = i+1; j <= min(i+C.w, C.T); j++) {
      float v = Dv.at<float>(i, j);
      if (v < dmin) {
        dmin = v;
        s = i;
        d = j;
      }
    }
  }

  LOG(INFO) << s << " " << d << " " << dmin;

  vector<int> path = {d};
  while (s > C.g) {
    path.insert(path.begin(), s);
    int b = Tv.at<int>(s, d);
    d = s;
    s = b;
    LOG(INFO) << d << " --> " << s;
  }
  END;

  START("Writing video");

  // For some reason, I can't encode the output with the same codec as the input.
  // Currently it's hardcoded to a version of h264 that works on OS X. See:
  // https://gist.github.com/takuma7/44f9ecb028ff00e2132e
  VideoWriter output(
    "out.mp4",
    CV_FOURCC('a','v','c','1'),
    input.get(CV_CAP_PROP_FPS),
    Size(input.get(CV_CAP_PROP_FRAME_WIDTH),
         input.get(CV_CAP_PROP_FRAME_HEIGHT)));

  if (!output.isOpened()) {
    cerr << "Output failed to open";
    exit(0);
  }

  for (int& idx : path) {
    LOG(INFO) << idx;
    output << frames[idx-1];
  }
  END;
}
