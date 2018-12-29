#include "mtcnn_detector.h"
#include "opencv_detector.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <memory>

using namespace std;
using namespace mtcnn;

string model_dir = "/sdcard/ryw/facies/caffe_full/";
const string pnet = "pnet";
const string rnet = "rnet";
const string onet = "onet";

int main(int argc, char** argv) {
  cout << argc << endl;
  if (argc > 1) {
    model_dir = argv[1];
  }

  std::shared_ptr<MTCNNDetector> detector = std::make_shared<OpencvDetector>();
  detector->Load({model_dir + pnet,
                 model_dir + rnet,
                 model_dir + onet});

  vector<FaceInfo> faces;
  string img_path;
  while (cin  >> img_path && img_path != "\0") {
    cv::Mat img = cv::imread(img_path.c_str());
    cout << "loading img " <<  img_path << " " << img.total() << endl;
    detector->Detect(img, faces);

    for (auto&& face: faces) {
      cv::Point pt1(face.bbox.xmin, face.bbox.ymin);
      cv::Point pt2(face.bbox.xmax, face.bbox.ymax);
      cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));

      auto& landmarks = face.landmark;
      for (int32_t i = 0; i < 5; i++) {
        cv::circle(img, cv::Point(landmarks[2*i], landmarks[i*2+1]), 1, cv::Scalar(0, 220, 100));
      }
    }

    cv::imwrite((img_path + ".result.jpg").c_str(), img);
    std::cout << "detect count " << faces.size() << std::endl;
  }
  return 0;
}
