#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<iostream>
#include "faceRec.h"

void instantiateVideoCapture(VideoCapture* capture){
  capture->set(CV_CAP_PROP_FRAME_WIDTH, 640);
  capture->set(CV_CAP_PROP_FRAME_HEIGHT, 480);
}

void updateSample(Ptr<FaceRecognizer> model, vector<Mat> images, vector<int> label){
  model->update(images, label);
}

void addImageAndLabelToVectors(Mat* frame, int label, vector<Mat>* images, vector<int>* labels){
  images->push_back(*frame);
  labels->push_back(label);
}

vector<Rect> findFacesInImage(Mat* frame, bool convert){
  if (convert){
    cvtColor(*frame, *frame, CV_BGR2GRAY);
  }
  equalizeHist(*frame, *frame);
  CascadeClassifier people_cascade;
  people_cascade.load("haarcascade_ped.xml");
  std::vector<Rect> people;
  people_cascade.detectMultiScale(*frame, people, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));
  return people;
}

FaceRec::FaceRec(){
  capture = new VideoCapture(0);
  model = createLBPHFaceRecognizer();
}

vector<Mat>* FaceRec::getImages(){
  return &images;
}

vector<Mat>* FaceRec::getNewImages(){
  return &newImages;
}

vector<int>* FaceRec::getLabels(){
  return &labels;
}

vector<int>* FaceRec::getNewLabels(){
  return &newLabels;
}

Ptr<FaceRecognizer>* FaceRec::getModel(){
  return &model;
}

VideoCapture* FaceRec::getCapture(){
  return capture;
}

FaceRec::~FaceRec(){
  delete capture;
}

int main(){
  FaceRec *faceRec = new FaceRec();

  instantiateVideoCapture(faceRec->getCapture());
  namedWindow("recognize", 1);

  for (int i = 1; i < 2; i++){
    faceRec->getImages()->push_back(imread(format("test-%d.jpg", i), 0));
    faceRec->getLabels()->push_back(1);
  }
  
  faceRec->getImages()->push_back(imread("sampDan1.jpg", 0));
  faceRec->getLabels()->push_back(2);
  
  Mat testSample = imread("sampDan2.jpg", 0);
  int testLabel = 2;
  
  Ptr<FaceRecognizer> model = *(faceRec->getModel());
  model->train(*(faceRec->getImages()), *(faceRec->getLabels()));

  while(true){
    Mat frame;
    Mat frameGray;
    *(faceRec->getCapture()) >> frame;
    cvtColor(frame, frameGray, CV_BGR2GRAY);
    int predictedLabel = model->predict(frameGray);
    cout << "Predicted class = " << predictedLabel << "/Actual class = " <<  testLabel << endl;
    imshow("recognize", frameGray);
    vector<Rect> faces = findFacesInImage(&frameGray, false);
  }
  return 0;
}
