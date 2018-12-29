#include <cmath>
#include <memory>
#include "mtcnn_detector.h"
#include "opencv_detector.h"

namespace mtcnn {

static const size_t kPNetMaxDetectNum = 5000;
static const int32_t kBatchSize = 128;

//mean & std image pre-process
static const float mean_val = 127.5f;
static const float std_val = 0.0078125f;

using std::min;
using std::max;
using cv::Rect;
using cv::Point;
using std::vector;
using std::string;

bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
  return a.bbox.score > b.bbox.score;
}

MTCNNDetector::MTCNNDetector()
              : min_img_size_(48)
              , pnet_threshold_(0.7)
              , rnet_threshold_(0.6)
              , onet_threshold_(0.99)
              , scale_factor_(0.709) {
}

int MTCNNDetector::Detect(const cv::Mat& img,
                          std::vector<FaceInfo>& faces) {
  return Detect(img, 3, faces);
}


std::vector<float> MTCNNDetector::GeneratePNetScales(int32_t w, int32_t h, int32_t size) {
  std::vector<float> scales;
  float scale = static_cast<float>(size) / min_img_size_;
  float min_wh = std::min(w, h) * scale;
  while (min_wh >= size) {
    scales.push_back(scale);
    min_wh *= scale_factor_;
    scale *= scale_factor_;
  }
  return scales;
}

float MTCNNDetector::IoU(
  float xmin, float ymin, float xmax, float ymax,
  float xmin_, float ymin_, float xmax_, float ymax_,
  bool is_iom) {
  const float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
  const float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
  if (iw <= 1e-8 || ih <= 1e-8) {
    return 0;
  }

  float ov = 0.0;
  const float s = iw * ih;
  if (is_iom) {
    ov = s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
  } else {
    ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
  }
  return ov;
}

void MTCNNDetector::BBoxRegression(std::vector<FaceInfo>& bboxes) {
  const int size = bboxes.size();
  for (int i = 0; i < size; ++i) {
    FaceBox& bbox = bboxes[i].bbox;
    float *bbox_reg = bboxes[i].bbox_reg;
    float w = bbox.xmax - bbox.xmin + 1;
    float h = bbox.ymax - bbox.ymin + 1;
    bbox.xmin += bbox_reg[0] * w;
    bbox.ymin += bbox_reg[1] * h;
    bbox.xmax += bbox_reg[2] * w;
    bbox.ymax += bbox_reg[3] * h;
  }
}

void MTCNNDetector::BBoxPadSquare(
  std::vector<FaceInfo>& bboxes,
  int width, int height) {
  for (int i = 0; i < bboxes.size(); ++i) {
    FaceBox& bbox = bboxes[i].bbox;
    const float w = bbox.xmax - bbox.xmin + 1;
    const float h = bbox.ymax - bbox.ymin + 1;
    const float side = h > w ? h : w;
    bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));
    bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
    bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
    bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
  }
}

void MTCNNDetector::BBoxPad(
  std::vector<FaceInfo>& bboxes,
  int width, int height) {
  for (int i = 0; i < bboxes.size(); ++i) {
    FaceBox& bbox = bboxes[i].bbox;
    bbox.xmin = round(max(bbox.xmin, 0.f));
    bbox.ymin = round(max(bbox.ymin, 0.f));
    bbox.xmax = round(min(bbox.xmax, width - 1.f));
    bbox.ymax = round(min(bbox.ymax, height - 1.f));
  }
}


std::vector<FaceInfo> MTCNNDetector::NMS(std::vector<FaceInfo>& bboxes,
                                         float thresh, char methodType) {
  std::vector<FaceInfo> bboxes_nms;
  if (bboxes.size() == 0U) {
    return bboxes_nms;
  }

  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);
  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  std::vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) {
      select_idx++;
    }
    if (select_idx >= num_bbox) {
      all_merged = true;
      break;
    }

    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;

    FaceBox select_bbox = bboxes[select_idx].bbox;
    float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
    float x1 = static_cast<float>(select_bbox.xmin);
    float y1 = static_cast<float>(select_bbox.ymin);
    float x2 = static_cast<float>(select_bbox.xmax);
    float y2 = static_cast<float>(select_bbox.ymax);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1)
        continue;

      FaceBox & bbox_i = bboxes[i].bbox;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
      if (w <= 0 || h <= 0)
        continue;

      float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
      float area_intersect = w * h;

      switch (methodType) {
        case 'u':
          if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
            mask_merged[i] = 1;
          break;
        case 'm':
          if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
            mask_merged[i] = 1;
          break;
        default:
          break;
      }
    }
  }
  return bboxes_nms;
}

std::vector<FaceInfo> MTCNNDetector::NMS_U(std::vector<FaceInfo>& bboxes, float thresh) {
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);
  for (const auto& bx : bboxes) {
    bool keep = true;
    for (int k = 0; k < static_cast<int>(bboxes_nms.size()) && keep; ++k) {
      const float overlap = IoU_U(bx.bbox, bboxes_nms[k].bbox);
      keep = overlap <= thresh;
    }
    if (keep) {
      bboxes_nms.push_back(bx);
    }
  }
  return bboxes_nms;
}

std::vector<FaceInfo> MTCNNDetector::NMS_M(std::vector<FaceInfo>& bboxes, float thresh) {
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);
  for (const auto& bbox : bboxes) {
    bool keep = true;
    for (int k = 0; k < static_cast<int>(bboxes_nms.size()) && keep; ++k) {
      const float overlap = IoU_M(bbox.bbox, bboxes_nms[k].bbox);
      keep = overlap <= thresh;
    }
    if (keep) {
      bboxes_nms.push_back(bbox);
    }
  }
  return bboxes_nms;
}

float MTCNNDetector::IoU_U(const FaceBox& a, const FaceBox& b) {
  const float iw = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin) + 1;
  const float ih = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin) + 1;
  if (iw <= 1e-8 || ih <= 1e-8) {
    return 0;
  }

  const float s = iw * ih;
  const float a_area = (a.xmax - a.xmin + 1) * (a.ymax - a.ymin + 1);
  const float b_area = (b.xmax - b.xmin + 1) * (b.ymax - b.ymin + 1);
  return s / (a_area + b_area - s);
}

float MTCNNDetector::IoU_M(const FaceBox& a, const FaceBox& b) {
  const float iw = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin) + 1;
  const float ih = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin) + 1;
  if (iw <= 1e-8 || ih <= 1e-8) {
    return 0;
  }

  const float s = iw * ih;
  const float a_area = (a.xmax - a.xmin + 1) * (a.ymax - a.ymin + 1);
  const float b_area = (b.xmax - b.xmin + 1) * (b.ymax - b.ymin + 1);
  return s / min(a_area, b_area);
}

/* static */
std::shared_ptr<MTCNNDetector> MTCNNDetector::CreateDetector() {
  // return std::make_shared<NcnnDetector>();
  return std::make_shared<OpencvDetector>();
}

void MTCNNDetector::DebugFaceInfo(const FaceInfo& face_info) {
  std::cout << "score " << face_info.bbox.score;
  std::cout << " bbox ["
    << face_info.bbox.xmin  << ","
    << face_info.bbox.ymin  << ","
    << face_info.bbox.xmax  << ","
    << face_info.bbox.ymax  << "] ";
  std::cout << "regbox [";
  for (int32_t i = 0; i < 4; i++) {
    std::cout << face_info.bbox_reg[i] << ",";
  }
  std::cout << "]" << std::endl;
}

}  // end of namespace mtcnn



