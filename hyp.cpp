#include <iostream>
#include <algorithm>
#include <limits>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "CycleTimer.h"

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::xfeatures2d;

class SectionTimer {
public:
  SectionTimer(string name) : name(name) {
    LOG(INFO) << name << "...";
    start = CycleTimer::currentSeconds();
  }

  ~SectionTimer() {
    LOG(INFO) << name << " done in " << CycleTimer::currentSeconds() - start << "s.";
  }

private:
  double start;
  string name;
};

class Constants {
public:
  int w = 32;
  int g = 4;
  int iw, ih;
  int T;
  int d;
  float tau_c, gamma;

  //int lam_s = 200;
  //int lam_a = 80;
  int lam_s = 100;
  int lam_a = 40;

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
                 cuda::SURF_CUDA& surf,
                 Ptr<cuda::DescriptorMatcher>& matcher,
                 vector<KeyPoint>& kp1, cuda::GpuMat& desc1_gpu,
                 vector<KeyPoint>& kp2, cuda::GpuMat& desc2_gpu)
{
  vector<DMatch> matches;
  matcher->match(desc1_gpu, desc2_gpu, matches);

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

  // If H is empty, then homography could not be found
  if (H.rows == 0) {
    return C.gamma;
  }

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

class VecSource : public IFrameSource {
public:
  VecSource(vector<Mat>* frames) {
    this->frames = frames;
    reset();
  }

  virtual void reset() {
    index = 0;
  }

  virtual Mat nextFrame() {
    if (index == frames->size()) {
      Mat m;
      return m;
    }

    Mat frame = (*frames)[index];
    index++;
    return frame;
  }

private:
  vector<Mat>* frames;
  int index;
};

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  double start;

  VideoCapture input("test.mp4");
  if (!input.isOpened()) {
    cerr << "Input failed to open";
    exit(0);
  }

  vector<Mat> frames;
  int minHessian = 400;
  vector<vector<KeyPoint>> kps;
  deque<cuda::GpuMat> features;
  cuda::SURF_CUDA surf(minHessian);

  bool set_constants = false;
  Constants C(0,0,0);

  Mat Cm = Mat::zeros(C.T+1, C.T+1, CV_32F);
  int T = 100;

  {
    SectionTimer timer("Loading frames");

    while(frames.size() < T) {
      Mat frame, frame_t, gray;
      cuda::GpuMat kp_gpu, feat_gpu, gpu_frame;
      if (!input.read(frame)) { break; }

      if (!set_constants) {
        C = Constants(frame.cols, frame.rows, T);
        set_constants = true;
      }

      transpose(frame, frame_t);
      cvtColor(frame_t, gray, CV_RGB2GRAY);
      frames.push_back(frame_t);
      gpu_frame.upload(gray);

      vector<KeyPoint> kp;
      surf(gpu_frame, cuda::GpuMat(), kp_gpu, feat_gpu);
      surf.downloadKeypoints(kp_gpu, kp);
      kps.push_back(kp);
      features.push_back(feat_gpu);


      int i = frames.size() - C.w;
      if (i >= 1) {
        Ptr<cuda::DescriptorMatcher> matcher =
          cuda::DescriptorMatcher::createBFMatcher();
        for (int j = i+1; j <= min(i+C.w, C.T); j++) {
          Cm.at<float>(i, j) =
            match_cost(C, surf, matcher, kps[i-1], features[0], kps[j-1], features[j-i-1]);
        }

        features[0].release();
        features.pop_front();
      }

      LOG(INFO) << i << " " << frames.size();
    }

    LOG(INFO) << "Loaded " << frames.size() << " frames.";
  }

  auto vel_cost = [&](int i, int j) {
    return std::min(powf((j - i) - C.v, 2), (float) C.tau_s);
  };

  auto acc_cost = [&](int h, int i, int j) {
    return std::min(powf((j - i) - (i - h), 2), (float) C.tau_a);
  };

  surf.releaseMemory();
  for (auto& feat : features) {
    feat.release();
  }

  vector<int> path;
  {
    SectionTimer timer("Finding path");
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

    path.push_back(d);
    while (s > C.g) {
      path.insert(path.begin(), s);
      int b = Tv.at<int>(s, d);
      d = s;
      s = b;
      //LOG(INFO) << d << " --> " << s;
    }
  }

  vector<Mat> optimalFrames;
  // for (int i = 0; i < frames.size() / 8; i++) {
  //   optimalFrames.push_back(frames[i * 8]);
  // }
  for (int idx : path) {
    LOG(INFO) << idx;
    optimalFrames.push_back(frames[idx-1]);
  }

#if CV_MAJOR_VERSION >= 3
  StabilizerBase* stabilizer;
  {
    TwoPassStabilizer* twoPass = new TwoPassStabilizer();
    twoPass->setMotionStabilizer(makePtr<GaussianMotionFilter>(15));
    stabilizer = twoPass;
  }

  Ptr<VecSource> source = makePtr<VecSource>(&optimalFrames);
  stabilizer->setFrameSource(source);

  Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(MM_HOMOGRAPHY);
  Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();

  // No GPU for now since I get OOM error for >1k frames
  // Ptr<KeypointBasedMotionEstimatorGpu> kbest = makePtr<KeypointBasedMotionEstimatorGpu>(est);
  Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
  kbest->setDetector(GFTTDetector::create(1000));

  kbest->setOutlierRejector(outlierRejector);
  stabilizer->setMotionEstimator(kbest);

  stabilizer->setRadius(15);
  stabilizer->setTrimRatio(0.1);
  stabilizer->setBorderMode(BORDER_REPLICATE);

  Ptr<WeightingDeblurer> deblurer = makePtr<WeightingDeblurer>();
  deblurer->setRadius(15);
  deblurer->setSensitivity(0.1);
  stabilizer->setDeblurer(deblurer);

  Ptr<IFrameSource> stabilizedFrames;
  stabilizedFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));
#else
  cout << "TODO: implement video stabilization for OpenCV 2.x";
  exit(0);
#endif

  {
    SectionTimer timer("Writing video");

    // For some reason, I can't encode the output with the same codec as the input.
    // Currently it's hardcoded to a version of h264 that works on OS X. See:
    // https://gist.github.com/takuma7/44f9ecb028ff00e2132e
    VideoWriter output;

    Mat frame;
    int count = 0;
    while (!(frame = stabilizedFrames->nextFrame()).empty()) {
      if (!output.isOpened()) {
        LOG(INFO) << frame.rows << " " << frame.cols << " "<< frame.channels();

        output.open(
          "out.mov",
          CV_FOURCC('a','v','c','1'),
          input.get(CV_CAP_PROP_FPS),
          frame.size());
      }

      output << frame;
      count++;
    }
  }
}
