#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("test.png");
	bitwise_not(img, img);

	Mat img_s = img;

	imshow("img", img); // show
	imshow("img_s", img_s);



	waitKey(0);
}