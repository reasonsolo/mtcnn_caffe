// Copyright(c) 2018, rongyi.com Inc.
// Author: ZHOU Lizhi <zhoulzh@rongyi.com>
#ifndef     MTCNN_OPENCV_DETECTOR_H_
#define     MTCNN_OPENCV_DETECTOR_H_

#include "mtcnn_detector.h"
#include "opencv2/opencv.hpp"

namespace mtcnn {

class OpencvDetector: public MTCNNDetector {
 public:
  OpencvDetector(): MTCNNDetector() {}

  virtual ~OpencvDetector() {}

  virtual bool Load(const std::vector<std::string>& model_paths) override;

  virtual int32_t Detect(const cv::Mat& img, int32_t stage, std::vector<FaceInfo>& faces) override;

 protected:
  virtual std::vector<FaceInfo> PNet(const cv::Mat& img, int32_t size,
                                     const std::vector<float>& scales);
  virtual std::vector<FaceInfo> RNet(const cv::Mat& img, int32_t size,
                                     const std::vector<FaceInfo>& prev_results);
  virtual std::vector<FaceInfo> ONet(const cv::Mat& img, int32_t size,
                                     const std::vector<FaceInfo>& prev_results);

  std::vector<FaceInfo> GenerateBBox(cv::Mat* confidence, cv::Mat* reg_box, int32_t cell_size,
                                     float scale, float thresh);

  cv::dnn::Net pnet_;
  cv::dnn::Net rnet_;
  cv::dnn::Net onet_;
};

}

#endif  //  MTCNN_OPENCV_DETECTOR_H_

