#include <iostream>
#include <vector>
#include <stack>
#include "opencv.h"
#include "SkinSegmenter.h"

using namespace std; 
using namespace cv;

int main() {
	VideoCapture video("e:/Video/VID_20180718_102220.mp4");
	if (!video.isOpened()) {
		cout << "fail to open!" << endl;
		return -1;
	}

	Mat image, targetImageHSV, rectImage;
	video >> image;
	SkinSegmenter skinSegmenter;
	
	skinSegmenter.setImage(image);
	Rect rect;
	skinSegmenter.getRect(rect);
	
	rectangle(skinSegmenter.origin, rect, Scalar(255, 0, 0), 2);
	imshow("origin", skinSegmenter.origin);

	//int channels[] = { 0,1 };
	//Mat dstHist;
	//int histSize = 200;
	//float histR[] = { 0,255 };
	//const float *histRange = histR;
	//rectImage = image(rect); //×ÓÍ¼ÏñÏÔÊ¾    
	////imshow("Sub Image", rectImage);
	//cvtColor(rectImage, targetImageHSV, CV_RGB2HSV);
	////imshow("targetImageHSV", targetImageHSV);
	//calcHist(&targetImageHSV, 2, channels, Mat(), dstHist, 1, &histSize, &histRange, true, false);
	//normalize(dstHist, dstHist, 0, 255, CV_MINMAX);
	////imshow("dstHist", dstHist);
	waitKey();
}
