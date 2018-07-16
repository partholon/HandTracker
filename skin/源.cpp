#include <iostream>
#include <vector>
#include <stack>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>

using namespace cv;
using namespace std;

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMakers(const cv::Mat& markerImage) {
		//convert to image of ints
		markerImage.convertTo(markers, CV_32S);
	}

	cv::Mat process(const cv::Mat &image) {
		//apply watershed
		cv::watershed(image, markers);
		return markers;
	}

	//return bin in the form of an image
	cv::Mat getSegmentation() {
		cv::Mat tmp;
		//all segment with label higher than 255 will be assigned value 255
		markers.convertTo(tmp, CV_8U);
		return tmp;
	}

	//return watershed inthe form of an image
	cv::Mat getWatersheds() {
		cv::Mat tmp;
		markers.convertTo(tmp, CV_8U, 255, 255);
		return tmp;
	}
};

void Seed_Filling(const cv::Mat& binImg, cv::Mat& labelImg, int& labelNum,
	int(&ymin)[20], int(&ymax)[20], int(&xmin)[20], int(&xmax)[20])   //种子填充法  
{
	if (binImg.empty() || binImg.type() != CV_8UC1)
		return;

	labelImg.release();
	binImg.convertTo(labelImg, CV_32FC1);
	int label = 2;
	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = labelImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				stack<pair<int, int>> neighborPixels;
				neighborPixels.push(pair<int, int>(j, i));     // 像素位置: <j,i>  
				ymin[label] = i;
				ymax[label] = i;
				xmin[label] = j;
				xmax[label] = j;
				while (!neighborPixels.empty())
				{
					std::pair<int, int> curPixel = neighborPixels.top(); //如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它  
					int curX = curPixel.first;
					int curY = curPixel.second;
					labelImg.at<int>(curY, curX) = label;

					neighborPixels.pop();

					// 1是二值图像中的边界
					if ((curX>0) && (curY>0) && (curX<(cols - 1)) && (curY<(rows - 1)))
					{
						if (labelImg.at<int>(curY - 1, curX) == 1)						//上
						{
							neighborPixels.push(std::pair<int, int>(curX, curY - 1));
							//ymin[label] = curY - 1;
						}
						if (labelImg.at<int>(curY + 1, curX) == 1)						//下
						{
							neighborPixels.push(std::pair<int, int>(curX, curY + 1));
							if ((curY + 1)>ymax[label])
								ymax[label] = curY + 1;
						}
						if (labelImg.at<int>(curY, curX - 1) == 1)						//左
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY));
							if ((curX - 1)<xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY, curX + 1) == 1)						//右
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY));
							if ((curX + 1)>xmax[label])
								xmax[label] = curX + 1;
						}
						if (labelImg.at<int>(curY - 1, curX - 1) == 1)					//左上
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX - 1)<xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY + 1, curX + 1) == 1)					//右下
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1));
							if ((curY + 1)>ymax[label])
								ymax[label] = curY + 1;
							if ((curX + 1)>xmax[label])
								xmax[label] = curX + 1;

						}
						if (labelImg.at<int>(curY + 1, curX - 1) == 1)					//左下
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1));
							if ((curY + 1)>ymax[label])
								ymax[label] = curY + 1;
							if ((curX - 1)<xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY - 1, curX + 1) == 1)					//右上
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX + 1)>xmax[label])
								xmax[label] = curX + 1;

						}
					}
				}
				label++;  // 没有重复的团，开始新的标签 
			}
		}
	}
	labelNum = label - 2;
}

int main() {
	vector<Mat> channels;
	Mat origin,bin,tmp;
	Mat Y, Cr, Cb;

	Mat tmpl = imread("bwz.jpg", CV_8UC1);
	origin = imread("2.jpg");
	//resize(origin, origin, Size(0,0), 0.15, 0.15, INTER_AREA);
	//namedWindow("origin");
	//imshow("origin", origin);

	//转换颜色空间
	//bin.create(resizeimg.rows, resizeimg.cols, CV_8UC1); //单通道灰度图
	cvtColor(origin, bin, CV_BGR2GRAY);
	origin.copyTo(tmp);
	cvtColor(tmp, tmp, CV_BGR2YCrCb);
	split(tmp, channels);
	Y = channels.at(0);
	Cr = channels.at(1);
	Cb = channels.at(2);
	//选择受亮度影响较小的Cb Cr作为依据
	for (int i = 1; i < Cb.rows - 1; i++) {
		uchar* pointCr = Cr.ptr<uchar>(i);
		uchar* pointCb = Cb.ptr<uchar>(i);
		uchar* pointbin = bin.ptr<uchar>(i);
		for (int j = 1; j < Cb.cols - 1; j++) {
			if ((pointCr[j] > 133) && (pointCr[j] < 173) && (pointCb[j] > 77) && (pointCb[j] < 127))
				pointbin[j] = 255;
			else
				pointbin[j] = 0;
		}
	}
	
	dilate(bin, bin, Mat());

	//watershed algorithm
	Mat fg;
	erode(bin, fg, Mat(), Point(-1, -1), 6);
	//Identify image pixels without objects
	Mat bg;
	dilate(bin, bg, Mat(), Point(-1, -1), 6);
	threshold(bg, bg, 1, 128, THRESH_BINARY_INV);
	//Show makers image
	Mat markers(bin.size(), CV_8U, Scalar(0));
	markers = fg + bg;
	//Create watershed segmentation object
	WatershedSegmenter segmenter;
	segmenter.setMakers(markers);
	segmenter.process(origin);
	Mat waterShed;
	waterShed = segmenter.getWatersheds();
	//get area
	threshold(waterShed, waterShed, 1, 1, THRESH_BINARY_INV);
	//imshow("watershed", waterShed);
	
	Mat labelImg;
	int label, ymin[20], ymax[20], xmin[20], xmax[20];
	Seed_Filling(waterShed, labelImg, label, ymin, ymax, xmin, xmax);
	
	////根据标记，对每块候选区就行缩放，并与模板比较
	//Size dsize = Size(tmpl.cols, tmpl.rows);
	//float simi[20];
	//for (int i = 0; i < label; i++)
	//{
	//	simi[i] = 1;
	//	if (((xmax[i] - xmin[i])>50) && ((ymax[i] - ymin[i]) > 50))
	//	{
	//		rectangle(bin, Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]), Scalar::all(255), 2, 8, 0);
	//		Mat rROI = Mat(dsize, CV_8UC1);
	//		resize(Cr(Rect(xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i])), rROI, dsize);
	//		Mat result;
	//		matchTemplate(rROI, tmpl, result, CV_TM_SQDIFF_NORMED);
	//		simi[i] = result.ptr<float>(0)[0];
	//		//cout << simi[i] << endl;
	//		imshow("temp", bin);
	//	}
	//}
	
	//统计一下区域中的肤色区域比例  --- 并没有排除噪点
	float fuseratio[20];
	for (int k = 0; k < label; k++)
	{
		fuseratio[k + 2] = 1;
		if (((xmax[k + 2] - xmin[k + 2])>50) && ((ymax[k + 2] - ymin[k + 2]) > 50))
		{
			int fusepoint = 0;
			for (int j = ymin[k + 2]; j < ymax[k + 2]; j++)
			{
				uchar* current = bin.ptr< uchar>(j);
				for (int i = xmin[k + 2]; i < xmax[k + 2]; i++)
				{
					if (current[i] == 255)
						fusepoint += 1;
				}
			}
			fuseratio[k + 2] = float(fusepoint) / ((xmax[k + 2] - xmin[k + 2])*(ymax[k + 2] - ymin[k + 2]));
			//cout << fuseratio[k + 2] << endl;
		}
	}

	//给符合阈值条件的位置画框
	for (int i = 0; i < label; i++)
	{
		if (/*(simi[i]<0.02) &&*/ (fuseratio[i] < 0.65))
			rectangle(origin, Point(xmin[i + 1], ymin[i + 2]), Point(xmax[i + 2], ymax[i + 2]), Scalar::all(255), 2, 8, 0);
	}
	
	//imshow("trans", bin);
	imshow("result", origin);
	waitKey();

	return 0;
}
