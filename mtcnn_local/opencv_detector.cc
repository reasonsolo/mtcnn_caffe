// Copyright(c) 2018, rongyi.com Inc.
// Author: ZHOU Lizhi <zhoulzh@rongyi.com>
#include "opencv_detector.h"
#include <iostream>

namespace mtcnn {

const std::string kCaffeAffix = ".caffemodel";
const std::string kProtoAffix = ".prototxt";

bool OpencvDetector::Load(const std::vector<std::string>& model_paths) {
  assert(model_paths.size() == 3);
  try {
    pnet_ = cv::dnn::readNetFromCaffe(model_paths[0] + kProtoAffix, model_paths[0] + kCaffeAffix);
    rnet_ = cv::dnn::readNetFromCaffe(model_paths[1] + kProtoAffix, model_paths[1] + kCaffeAffix);
    onet_ = cv::dnn::readNetFromCaffe(model_paths[2] + kProtoAffix, model_paths[2] + kCaffeAffix);
  } catch (const std::exception& e) {
    std::cout << "cannot load model from " << model_paths[0]  << ", " << model_paths[1] << ", " << model_paths[2] << std::endl;
    return false;
  }
  return true;
}

std::vector<FaceInfo> OpencvDetector::PNet(const cv::Mat& img, int32_t size,
                                           const std::vector<float>& scales) {
  const int32_t w = img.cols;
  const int32_t h = img.rows;
  std::vector<FaceInfo> results;
  for (auto& scale: scales) {
    const int32_t sw = std::ceil(scale* w);
    const int32_t sh = std::ceil(scale* h);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(sw, sh), 0, 0, cv::INTER_LINEAR);

    cv::Mat input_blob = cv::dnn::blobFromImage(resized, kImgStdVal, cv::Size(),
                                                cv::Scalar(kImgMeanVal, kImgMeanVal, kImgMeanVal), false);

    pnet_.setInput(input_blob, "data");
    const std::vector<cv::String>  targets_node{"conv4-2","prob1"};
    std::vector<cv::Mat> targets_blobs;
    pnet_.forward(targets_blobs, targets_node);

    cv::Mat reg = targets_blobs[0];
    cv::Mat prob = targets_blobs[1];

    auto bboxes = GenerateBBox(&prob, &reg, size, scale, pnet_threshold_);
    auto nms_bboxes = NMS_U(bboxes, 0.5);
    results.insert(results.end(), nms_bboxes.begin(), nms_bboxes.end());
  }
  return results;
}

std::vector<FaceInfo> OpencvDetector::RNet(const cv::Mat& img, int32_t size,
                                           const std::vector<FaceInfo>& prev_results) {
  std::vector<FaceInfo> results;
  if (prev_results.empty()) {
    return results;
  }

  std::vector<cv::Mat> inputs;
  for (auto&& face_info: prev_results) {
    const FaceBox& box = face_info.bbox;
    cv::Mat roi = img(cv::Rect(cv::Point((int)box.xmin, (int)box.ymin),
                               cv::Point((int)box.xmax, (int)box.ymax))).clone();
    cv::resize(roi, roi, cv::Size(size, size));
    inputs.push_back(roi);
  }

  cv::Mat blob_input = cv::dnn::blobFromImages(inputs, kImgStdVal, cv::Size(),
                                               cv::Scalar(kImgMeanVal, kImgMeanVal, kImgMeanVal), false);
  rnet_.setInput(blob_input, "data");
  std::vector<cv::Mat> targets_blobs;
  std::vector<cv::String> targets_node{"conv5-2","prob1"};
  rnet_.forward(targets_blobs, targets_node);

  cv::Mat* confidence = &targets_blobs[1];
  cv::Mat* reg_box = &targets_blobs[0];

  const float* confidence_data = reinterpret_cast<float*>(confidence->data);
  const float* reg_data = reinterpret_cast<float*>(reg_box->data);

  for (int32_t i = 0; i < prev_results.size(); i++) {
    float score =  confidence_data[2 * i + 1];
    if (score >= rnet_threshold_) {
      FaceInfo face_info = prev_results[i];
      face_info.bbox.score = score;
      for (int32_t j = 0; j < 4; j++) {
        face_info.bbox_reg[j] = reg_data[4 * i + j];
      }
      results.push_back(face_info);
      // DebugFaceInfo(face_info);
    }
  }

  return results;
}

std::vector<FaceInfo> OpencvDetector::ONet(const cv::Mat& img, int32_t size,
                                           const std::vector<FaceInfo>& prev_results) {
  std::vector<FaceInfo> results;
  if (prev_results.empty()) {
    return results;
  }

  std::vector<cv::Mat> inputs;
  for (auto&& face_info: prev_results) {
    const FaceBox& box = face_info.bbox;
    cv::Mat roi = img(cv::Rect(cv::Point((int)box.xmin, (int)box.ymin),
                               cv::Point((int)box.xmax, (int)box.ymax))).clone();
    cv::resize(roi, roi, cv::Size(size, size));
    inputs.push_back(roi);
  }

  cv::Mat blob_input = cv::dnn::blobFromImages(inputs, kImgStdVal, cv::Size(),
                                               // cv::Scalar(kImgMeanVal, kImgMeanVal, kImgMeanVal),
                                               cv::Scalar(0, 0, 0),
                                               false);
  onet_.setInput(blob_input, "data");
  std::vector<cv::Mat> targets_blobs;
  std::vector<cv::String> targets_node{"conv6-2","conv6-3","prob1"};
  onet_.forward(targets_blobs, targets_node);

  cv::Mat* reg_box = &targets_blobs[0];
  cv::Mat* reg_landmark = &targets_blobs[1];
  cv::Mat* confidence = &targets_blobs[2];

  const float* confidence_data = reinterpret_cast<float*>(confidence->data);
  const float* reg_data = reinterpret_cast<float*>(reg_box->data);
  const float* landmark_data = reinterpret_cast<float*>(reg_landmark->data);

  for (int32_t i = 0; i < prev_results.size(); i++) {
    float score =  confidence_data[2 * i + 1];
    if (score >= rnet_threshold_) {
      FaceInfo face_info = prev_results[i];
      face_info.bbox.score = score;
      for (int32_t j = 0; j < 4; j++) {
        face_info.bbox_reg[j] = reg_data[4 * i + j];
      }
      float w = face_info.bbox.xmax - face_info.bbox.xmin + 1.f;
      float h = face_info.bbox.ymax - face_info.bbox.ymin + 1.f;
      for (int k = 0; k < 5; ++k) {
        face_info.landmark[2 * k] = landmark_data[10 * i + 2 * k] * w + face_info.bbox.xmin;
        face_info.landmark[2 * k + 1] = landmark_data[10 * i + 2 * k + 1] * h + face_info.bbox.ymin;
      }
      results.push_back(face_info);
      // DebugFaceInfo(face_info);
    }
  }
  return results;
}

/**
 * Here assume confidence format is 1 * 2 * h * w (4 dimension)
 * 1: means currently the batch size is 1
 * 2: [0]=>indicate the probability of 1, [1]=>indicate the probability of 0
 * h * w: there are h * w boxes in currently image
 **/
std::vector<FaceInfo> OpencvDetector::GenerateBBox(cv::Mat* confidence, cv::Mat* reg_box,
                                                   int32_t cell_size, float scale, float thresh) {
  std::vector<FaceInfo> results;

  const int feature_map_w = confidence->size[3];
  const int feature_map_h = confidence->size[2];
  const int spatical_size = feature_map_w * feature_map_h;

  const float* confidence_data = (float*)(confidence->data);
  confidence_data += spatical_size;
  const float* reg_data = (float*)(reg_box->data);

  const float t = 1 - thresh;
  for (int i = 0; i < spatical_size; i++) {
    if (confidence_data[i] <= t) {
      const int y = i / feature_map_w;
      const int x = i - feature_map_w * y;
      FaceInfo face_info;
      FaceBox& faceBox = face_info.bbox;
      faceBox.xmin = (float)(x * kPNetStride) / scale;
      faceBox.ymin = (float)(y * kPNetStride) / scale;
      faceBox.xmax = (float)(x * kPNetStride + cell_size - 1.f) / scale;
      faceBox.ymax = (float)(y * kPNetStride + cell_size - 1.f) / scale;
      face_info.bbox_reg[0] = reg_data[i];
      face_info.bbox_reg[1] = reg_data[i + spatical_size];
      face_info.bbox_reg[2] = reg_data[i + 2 * spatical_size];
      face_info.bbox_reg[3] = reg_data[i + 3 * spatical_size];
      faceBox.score = 1 - confidence_data[i];
      results.push_back(face_info);
    }
  }
  return results;
}

int32_t OpencvDetector::Detect(const cv::Mat& img, const int stage,
                               std::vector<FaceInfo>& faces) {
  std::vector<FaceInfo> pnet_res;
  std::vector<FaceInfo> rnet_res;
  std::vector<FaceInfo> onet_res;

  const int32_t img_w = img.cols;
  const int32_t img_h = img.rows;

  if (stage >= 1){
    pnet_res = PNet(img, kPNetSize, GeneratePNetScales(img_w, img_h, kPNetSize));
    pnet_res = NMS_U(pnet_res, 0.7f);
    BBoxRegression(pnet_res);
    BBoxPadSquare(pnet_res, img_w, img_h);
  }

  if (stage >= 2) {
    rnet_res = RNet(img, kRNetSize, pnet_res);
    rnet_res = NMS_M(rnet_res, 0.4f);
    BBoxRegression(rnet_res);
    BBoxPadSquare(rnet_res, img_w, img_h);
  }

  if (stage >= 3) {
    onet_res = ONet(img, kONetSize, rnet_res);
    BBoxRegression(onet_res);          // refine bbox first
    onet_res = NMS_M(onet_res, 0.4f);  // then do NMS
    BBoxPad(onet_res, img_w, img_h);
  }

  if (stage == 1){
    faces.swap(pnet_res);
  } else if (stage == 2){
    faces.swap(rnet_res);
  } else{
    faces.swap(onet_res);
  }
  return faces.size();
}

}
