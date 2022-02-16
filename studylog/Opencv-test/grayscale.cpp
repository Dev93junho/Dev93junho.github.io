#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img_color;
	img_color = imread("test.png");

	if (img_color.empty())
	{
		cout << "Do not Read image file" << endl;
		return -1;
	}
	
	namedWindow("init image");

	imshow("init image", img_color); // show

	waitKey(0); // wait til other input

	Mat img_gray;

	//COLOR.BGR2GRAY
	cvtColor(img_color, img_gray, COLOR_BGR2GRAY); // Image Convert  "BGR to GRAY color"

	imshow("GrayScale", img_gray); // Print the Image
	imwrite("gray.jpg",img_gray); // Save the Image
	waitKey(0);
	destroyAllWindows();
}