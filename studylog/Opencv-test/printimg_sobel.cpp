#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int, char) {
	Mat img = imread("test.png");
	Sobel(img, img, img.depth(), 1, 0); // print sobel edge
	namedWindow("img", 0);

	imshow("img", img); // window open

	// â�� �ٷ� ������ �ʰ� �ܺ� �Է��� ��ٸ��� ����
	waitKey(0);  // 0�� ���Ѵ�, �� �̻��� ms ����


	return 0;
}
