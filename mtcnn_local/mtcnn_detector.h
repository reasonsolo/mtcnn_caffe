#ifndef THIRDPARTY_MTCNN_MTCNN_DETECTOR_H_
#define THIRDPARTY_MTCNN_MTCNN_DETECTOR_H_

#include <vector>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"

namespace mtcnn {

struct FaceBox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;

  float score;
};

struct FaceInfo {
  float bbox_reg[4];
  float landmark_reg[10];

  float landmark[10];
  FaceBox bbox;
};

class MTCNNDetector {

 public:
  MTCNNDetector();
  // MTCNNDetector(const std::vector<float>& thresholds, int32_t min_img_size, float scale_factor);
  virtual ~MTCNNDetector() {}

  virtual bool Load(const std::vector<std::string>& model_paths) = 0;

  virtual int32_t Detect(const cv::Mat& img, std::vector<FaceInfo>& faces);

  void min_img_size(int32_t size) { min_img_size_ = size; }

  static std::shared_ptr<MTCNNDetector> CreateDetector();

 protected:

  std::vector<float> GeneratePNetScales(int32_t w, int32_t h, int32_t size);

  virtual int Detect(const cv::Mat& img, const int stage, std::vector<FaceInfo>& faces) = 0;

  void DebugFaceInfo(const FaceInfo& face_info);

  void BBoxRegression(std::vector<FaceInfo>& bboxes);
  void BBoxPadSquare(std::vector<FaceInfo>& bboxes, int width, int height);
  void BBoxPad(std::vector<FaceInfo>& bboxes, int width, int height);
  std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
  std::vector<FaceInfo> NMS_U(std::vector<FaceInfo>& bboxes, float thresh);
  std::vector<FaceInfo> NMS_M(std::vector<FaceInfo>& bboxes, float thresh);
  float IoU_U(const FaceBox& a, const FaceBox& b);
  float IoU_M(const FaceBox& a, const FaceBox& b);
  float IoU(float xmin, float ymin, float xmax, float ymax,
            float xmin_, float ymin_, float xmax_, float ymax_,
            bool is_iom = false);

  int32_t min_img_size_;
  float pnet_threshold_;
  float rnet_threshold_;
  float onet_threshold_;
  float scale_factor_;

  const int32_t kPNetSize = 12;
  const int32_t kRNetSize = 24;
  const int32_t kONetSize = 48;

  const float kImgMeanVal = 127.5f;
  const float kImgStdVal = 1. / 255;

  const float kPNetStride = 2;

  static constexpr const char* kPnetModel = "pnet";
  static constexpr const char* kRnetModel = "rnet";
  static constexpr const char* kOnetModel = "onet";
};
}  // end of namespace mtcnn

#endif  // end of THIRDPARTY_MTCNN_MTCNN_DETECTOR_H_
