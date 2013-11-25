#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/objdetect/objdetect.hpp>
using namespace std;
using namespace cv;
#ifndef FACEREC_H_
#define FACEREC_H_
class FaceRec {
  public:
    FaceRec();
    vector<Mat>* getImages();
    vector<Mat>* getNewImages();
    vector<int>* getLabels();
    vector<int>* getNewLabels();
    Ptr<FaceRecognizer>* getModel();
    VideoCapture* getCapture();
    ~FaceRec();

  private:
    vector<Mat> images;
    vector<Mat> newImages;
    vector<int> labels;
    vector<int> newLabels;
    Ptr<FaceRecognizer> model;
    VideoCapture *capture;
};
#endif
