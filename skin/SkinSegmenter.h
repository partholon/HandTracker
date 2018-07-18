#pragma once
#include "opencv.h"
#include <vector>
#include <stack>

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMakers(const cv::Mat& markerImage);
	cv::Mat process(const cv::Mat &image);
	//return bin in the form of an image
	cv::Mat getSegmentation();
	//return watershed inthe form of an image
	cv::Mat getWatersheds();
};


class SkinSegmenter {
private:
	void eightConnections(const cv::Mat& binImg, int& labelNum,
		std::vector<int>& ymin, std::vector<int>& ymax, std::vector<int>& xmin, std::vector<int>& xmax);

	cv::Mat getWatershed();

public:
	SkinSegmenter(const char* filename);

	~SkinSegmenter();

	bool getRect(cv::Rect& rect);

private:
	cv::Mat binImage, tmp;
	cv::Mat Y, Cr, Cb;
	cv::Mat frontGround, backGround, waterShed;
	std::vector<cv::Mat> channels;
	int label;
	std::vector<int>ymin, ymax, xmin, xmax;

public:
	cv::Mat origin;
};