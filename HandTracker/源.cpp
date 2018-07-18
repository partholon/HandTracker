#include "../skin/opencv.h"
#include "../skin/SkinSegmenter.h"
#include <iostream>      

using namespace cv;
using namespace std;

int main(int argc, char*argv[])
{
	Mat image;
	Mat rectImage;
	Mat imageCopy; //绘制矩形框时用来拷贝原图的图像    
	bool leftButtonDownFlag = false; //左键单击后视频暂停播放的标志位    
	bool stop = false;
	Point originalPoint; //矩形框起点    
	Point processPoint; //矩形框终点    

	Mat targetImageHSV;
	int histSize = 200;
	float histR[] = { 0,255 };
	const float *histRange = histR;
	int channels[] = { 0,1 };
	Mat dstHist;
	Rect rect;
	vector<Point> pt; //保存目标轨迹  

	VideoCapture video("e:/video/VID_20180718_102220.mp4");
	if (!video.isOpened()) {
		cout << "fail to open!" << endl;
		return -1;
	}

	double fps = video.get(CV_CAP_PROP_FPS); //获取视频帧率    
	double pauseTime = 1000 / fps; //两幅画面中间间隔    
	namedWindow("跟踪木头人", 0);
	//setMouseCallback("跟踪木头人", onMouse);
	video >> image;

	SkinSegmenter skinSegmenter;
	skinSegmenter.setImage(image);
	skinSegmenter.getRect(rect);
	rectImage = image(rect); //子图像显示    
	cvtColor(rectImage, targetImageHSV, CV_RGB2HSV);
	calcHist(&targetImageHSV, 2, channels, Mat(), dstHist, 1, &histSize, &histRange, true, false);
	normalize(dstHist, dstHist, 0, 255, CV_MINMAX);
	
	//int resetFrameNum = 1; //重置帧位
	while (true)
	{
		if (!video.read(image)) 		{
			break;
		}
		if (!image.data && waitKey(pauseTime) == 27)  //图像为空或Esc键按下退出播放    
		{
			break;
		}

		//需要解决窗口缩放带来的位置变动
		//int frame_num = video.get(CV_CAP_PROP_POS_FRAMES);
		//if (frame_num - resetFrameNum == 10) {
		//	skinSegmenter.setImage(image);
		//	skinSegmenter.getRect(rect);
		//	rectImage = image(rect); //子图像显示    
		//	cvtColor(rectImage, targetImageHSV, CV_RGB2HSV);
		//	calcHist(&targetImageHSV, 2, channels, Mat(), dstHist, 1, &histSize, &histRange, true, false);
		//	normalize(dstHist, dstHist, 0, 255, CV_MINMAX);
		//	resetFrameNum = frame_num;
		//	std::cout << "Frame Num : " << frame_num << std::endl;
		//}

		Mat imageHSV;
		Mat calcBackImage;
		cvtColor(image, imageHSV, CV_RGB2HSV);
		calcBackProject(&imageHSV, 2, channels, dstHist, calcBackImage, &histRange);  //反向投影  
		TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.001);
		CamShift(calcBackImage, rect, criteria);
		Mat imageROI = imageHSV(rect);   //更新模板             
		targetImageHSV = imageHSV(rect);
		calcHist(&imageROI, 2, channels, Mat(), dstHist, 1, &histSize, &histRange);
		normalize(dstHist, dstHist, 0.0, 1.0, NORM_MINMAX);   //归一化  
		rectangle(image, rect, Scalar(255, 0, 0), 3);    //目标绘制    
		pt.push_back(Point(rect.x + rect.width / 2, rect.y + rect.height / 2));
		for (int i = 0; i<pt.size() - 1; i++)
		{
			line(image, pt[i], pt[i + 1], Scalar(0, 255, 0), 2.5);
		}

		imshow("跟踪木头人", image);
		waitKey(10);
	}
	return 0;
}