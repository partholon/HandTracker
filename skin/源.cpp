#include <iostream>
#include <vector>
#include <stack>
#include "opencv.h"
#include "SkinSegmenter.h"

using namespace std; 
using namespace cv;

int main() {
	SkinSegmenter skinSegmenter("low.jpg");

	Rect rect;
	skinSegmenter.getRect(rect);
	

}
