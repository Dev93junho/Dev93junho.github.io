#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int, char) {
	//Declaration and at the same time created
	Mat mtx(3, 3, CV_32F); //make the 3 x 3, 32bit, floating-point Matrix
	Mat cmtx(10, 1, CV_64FC2); // make the 10 x 1, 64bit, 2-channel, floating-point Matrix, 10-element complex vector
	Mat img(Size(5, 3), CV_8UC3); // make a 3-channel, color image with [ width 5, height 3 ] ==> Size function, 8bit - unsigned character.

	Mat mtx2;
	mtx2 = Mat(3, 3, CV_32F);

	Mat cmtx2;
	cmtx2 = Mat(10, 1, CV_64FC1);
}
