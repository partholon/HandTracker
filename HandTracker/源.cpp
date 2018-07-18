#include "../skin/opencv.h"
#include "../skin/SkinSegmenter.h"
#include <iostream>      

using namespace cv;
using namespace std;

int main(int argc, char*argv[])
{
	Mat image;
	Mat rectImage;
	Mat imageCopy; //���ƾ��ο�ʱ��������ԭͼ��ͼ��    
	bool leftButtonDownFlag = false; //�����������Ƶ��ͣ���ŵı�־λ    
	bool stop = false;
	Point originalPoint; //���ο����    
	Point processPoint; //���ο��յ�    

	Mat targetImageHSV;
	int histSize = 200;
	float histR[] = { 0,255 };
	const float *histRange = histR;
	int channels[] = { 0,1 };
	Mat dstHist;
	Rect rect;
	vector<Point> pt; //����Ŀ��켣  

	VideoCapture video("e:/video/VID_20180718_102220.mp4");
	if (!video.isOpened()) {
		cout << "fail to open!" << endl;
		return -1;
	}

	double fps = video.get(CV_CAP_PROP_FPS); //��ȡ��Ƶ֡��    
	double pauseTime = 1000 / fps; //���������м���    
	namedWindow("����ľͷ��", 0);
	//setMouseCallback("����ľͷ��", onMouse);
	video >> image;

	SkinSegmenter skinSegmenter;
	skinSegmenter.setImage(image);
	skinSegmenter.getRect(rect);
	rectImage = image(rect); //��ͼ����ʾ    
	cvtColor(rectImage, targetImageHSV, CV_RGB2HSV);
	calcHist(&targetImageHSV, 2, channels, Mat(), dstHist, 1, &histSize, &histRange, true, false);
	normalize(dstHist, dstHist, 0, 255, CV_MINMAX);
	
	//int resetFrameNum = 1; //����֡λ
	while (true)
	{
		if (!video.read(image)) 		{
			break;
		}
		if (!image.data && waitKey(pauseTime) == 27)  //ͼ��Ϊ�ջ�Esc�������˳�����    
		{
			break;
		}

		//��Ҫ����������Ŵ�����λ�ñ䶯
		//int frame_num = video.get(CV_CAP_PROP_POS_FRAMES);
		//if (frame_num - resetFrameNum == 10) {
		//	skinSegmenter.setImage(image);
		//	skinSegmenter.getRect(rect);
		//	rectImage = image(rect); //��ͼ����ʾ    
		//	cvtColor(rectImage, targetImageHSV, CV_RGB2HSV);
		//	calcHist(&targetImageHSV, 2, channels, Mat(), dstHist, 1, &histSize, &histRange, true, false);
		//	normalize(dstHist, dstHist, 0, 255, CV_MINMAX);
		//	resetFrameNum = frame_num;
		//	std::cout << "Frame Num : " << frame_num << std::endl;
		//}

		Mat imageHSV;
		Mat calcBackImage;
		cvtColor(image, imageHSV, CV_RGB2HSV);
		calcBackProject(&imageHSV, 2, channels, dstHist, calcBackImage, &histRange);  //����ͶӰ  
		TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.001);
		CamShift(calcBackImage, rect, criteria);
		Mat imageROI = imageHSV(rect);   //����ģ��             
		targetImageHSV = imageHSV(rect);
		calcHist(&imageROI, 2, channels, Mat(), dstHist, 1, &histSize, &histRange);
		normalize(dstHist, dstHist, 0.0, 1.0, NORM_MINMAX);   //��һ��  
		rectangle(image, rect, Scalar(255, 0, 0), 3);    //Ŀ�����    
		pt.push_back(Point(rect.x + rect.width / 2, rect.y + rect.height / 2));
		for (int i = 0; i<pt.size() - 1; i++)
		{
			line(image, pt[i], pt[i + 1], Scalar(0, 255, 0), 2.5);
		}

		imshow("����ľͷ��", image);
		waitKey(10);
	}
	return 0;
}