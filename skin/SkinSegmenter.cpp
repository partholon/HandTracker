#include "SkinSegmenter.h"

using namespace std;
using namespace cv;

void WatershedSegmenter::setMakers(const Mat& markerImage) {
	//convert to image of ints
	markerImage.convertTo(markers, CV_32S);
}

Mat WatershedSegmenter::process(const Mat &image) {
	//apply watershed
	watershed(image, markers);
	return markers;
}

//return bin in the form of an image
Mat WatershedSegmenter::getSegmentation() {
	Mat tmp;
	//all segment with label higher than 255 will be assigned value 255
	markers.convertTo(tmp, CV_8U);
	return tmp;
}

//return watershed inthe form of an image
Mat WatershedSegmenter::getWatersheds() {
	Mat tmp;
	markers.convertTo(tmp, CV_8U, 255, 255);
	return tmp;
}

SkinSegmenter::SkinSegmenter() {

}

SkinSegmenter::~SkinSegmenter() {

}

void SkinSegmenter::setImage(const cv::Mat image) {
	origin = image;
	//resize(origin, origin, Size(0, 0), 0.15, 0.15, INTER_AREA);
	//imshow("origin", origin);

	//转换颜色空间
	cvtColor(origin, binImage, CV_BGR2GRAY);
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
		uchar* pointbin = binImage.ptr<uchar>(i);
		for (int j = 1; j < Cb.cols - 1; j++) {
			if ((pointCr[j] > 135) && (pointCr[j] < 180) && (pointCb[j] > 85) && (pointCb[j] < 135))
				pointbin[j] = 255;
			else
				pointbin[j] = 0;
		}
	}
}

//八向联通确定边界
void SkinSegmenter::eightConnections(const Mat& binImg, int& labelNum,
	vector<int>& ymin, vector<int>& ymax, vector<int>& xmin, vector<int>& xmax) {

	if (binImg.empty() || binImg.type() != CV_8UC1)
		return;

	Mat labelImg;
	binImg.convertTo(labelImg, CV_32SC1);

	int label = 2;
	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows - 1; i++) {
		int* data = labelImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++) {
			if (*(data + j) == 1) {
				stack<pair<int, int>> neighborPixels;
				neighborPixels.push(pair<int, int>(j, i));
				ymin.push_back(i);
				ymax.push_back(i);
				xmin.push_back(j);
				xmax.push_back(j);
				while (!neighborPixels.empty()) {
					pair<int, int> curPixel = neighborPixels.top();
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
							if ((curY + 1)>ymax[label - 2])
								ymax[label - 2] = curY + 1;
						}
						if (labelImg.at<int>(curY, curX - 1) == 1)						//左
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY));
							if ((curX - 1)<xmin[label - 2])
								xmin[label - 2] = curX - 1;
						}
						if (labelImg.at<int>(curY, curX + 1) == 1)						//右
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY));
							if ((curX + 1)>xmax[label - 2])
								xmax[label - 2] = curX + 1;
						}
						if (labelImg.at<int>(curY - 1, curX - 1) == 1)					//左上
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1));
							//ymin[label-2] = curY - 1;
							if ((curX - 1)<xmin[label - 2])
								xmin[label - 2] = curX - 1;
						}
						if (labelImg.at<int>(curY + 1, curX + 1) == 1)					//右下
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1));
							if ((curY + 1)>ymax[label - 2])
								ymax[label - 2] = curY + 1;
							if ((curX + 1)>xmax[label - 2])
								xmax[label - 2] = curX + 1;

						}
						if (labelImg.at<int>(curY + 1, curX - 1) == 1)					//左下
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1));
							if ((curY + 1)>ymax[label - 2])
								ymax[label - 2] = curY + 1;
							if ((curX - 1)<xmin[label - 2])
								xmin[label - 2] = curX - 1;
						}
						if (labelImg.at<int>(curY - 1, curX + 1) == 1)					//右上
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1));
							//ymin[label-2] = curY - 1;
							if ((curX + 1)>xmax[label - 2])
								xmax[label - 2] = curX + 1;

						}
					}
				}
				label++;
			}
		}
	}
	labelNum = label - 2;
}

Mat SkinSegmenter::getWatershed() {
	//this->binImage
	//dilate(binImage, binImage, Mat());
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));   // 开运算去除噪点
	morphologyEx(binImage, binImage, MORPH_OPEN, element);

	//watershed algorithm
	erode(binImage, frontGround, Mat(), Point(-1, -1), 6);
	//Identify image pixels without objects
	dilate(binImage, backGround, Mat(), Point(-1, -1), 6);
	threshold(backGround, backGround, 1, 128, THRESH_BINARY_INV);
	//Show makers image
	Mat markers(binImage.size(), CV_8U, Scalar(0));
	markers = frontGround + backGround;
	//Create watershed segmentation object
	WatershedSegmenter segmenter;
	segmenter.setMakers(markers);
	segmenter.process(origin);
	Mat waterShed;
	waterShed = segmenter.getWatersheds();
	//imshow("watershed", waterShed);
	//Get area
	threshold(waterShed, waterShed, 1, 1, THRESH_BINARY_INV);
	//imshow("watershed", waterShed);

	return waterShed;
}

bool SkinSegmenter::getRect(Rect& rect) {
	waterShed = this->getWatershed();
	
	this->eightConnections(waterShed, label, ymin, ymax, xmin, xmax);
	
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
		fuseratio[k] = 1;
		if (((xmax[k] - xmin[k])>50) && ((ymax[k] - ymin[k]) > 50))
		{
			int fusepoint = 0;
			for (int j = ymin[k]; j < ymax[k]; j++)
			{
				uchar* current = binImage.ptr< uchar>(j);
				for (int i = xmin[k]; i < xmax[k]; i++)
				{
					if (current[i] == 255)
						fusepoint += 1;
				}
			}
			fuseratio[k] = float(fusepoint) / ((xmax[k] - xmin[k])*(ymax[k] - ymin[k]));
			//cout << fuseratio[k] << endl;
		}
	}

	//给符合阈值条件的位置画框
	for (int i = 0; i < label; i++)
	{
		if (/*(simi[i]<0.02) &&*/ (fuseratio[i] < 0.8)) { //0.65->0.8
			rect = Rect(Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]));
			return true;
		}
	}

	return false;
}