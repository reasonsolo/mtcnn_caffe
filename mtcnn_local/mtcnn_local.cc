#include "mtcnn_detector.h"
#include "opencv_detector.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <memory>

using namespace std;
using namespace mtcnn;

string model_dir = "./models/";
string store_dir = "./faces/";
const string pnet = "pnet";
const string rnet = "rnet";
const string onet = "onet";

int main(int argc, char** argv) {
  if (argc > 1) {
    model_dir = argv[1];
  }
  if (argc > 2) {
    store_dir = argv[2];
  }

  std::shared_ptr<MTCNNDetector> detector = std::make_shared<OpencvDetector>();
  detector->Load({model_dir + pnet,
                 model_dir + rnet,
                 model_dir + onet});

  vector<FaceInfo> faces;
  string img_path;
  int i = 0;
  while (cin  >> img_path && img_path != "exit") {
    if (img_path.size() == 0) {
      continue;
    }
    cv::Mat img = cv::imread(img_path.c_str());
    detector->Detect(img, faces);
    for (auto&& face: faces) {
      //cv::Point pt1(face.bbox.xmin, face.bbox.ymin);
      //cv::Point pt2(face.bbox.xmax, face.bbox.ymax);
      //cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));

      // auto& landmarks = face.landmark;
      // for (int32_t i = 0; i < 5; i++) {
      //   cv::circle(img, cv::Point(landmarks[2*i], landmarks[i*2+1]), 1, cv::Scalar(0, 220, 100));
      // }
      cout << "[" << face.bbox.xmin << "," << face.bbox.ymin << "," << face.bbox.xmax << "," << face.bbox.ymax << "],";
      // auto cropped = img(cv::Rect(face.bbox.xmin, face.bbox.ymin,  face.bbox.xmax-face.bbox.xmin, face.bbox.ymax-face.bbox.ymin));
      // cv::imwrite((store_dir + std::to_string(++i) + ".jpg").c_str(), cropped);
    }
    cout << endl;
  }
  return 0;
}

