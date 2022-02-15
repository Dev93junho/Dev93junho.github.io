#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int, char) {
	Mat img = imread("test.png");
	Sobel(img, img, img.depth(), 1, 0); // print sobel edge
	namedWindow("img", 0);

	imshow("img", img); // window open

	// 창이 바로 닫히지 않게 외부 입력을 기다리는 구문
	waitKey(0);  // 0은 무한대, 그 이상은 ms 단위


	return 0;
}
