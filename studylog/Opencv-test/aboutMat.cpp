#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;



int main(int, char)
{
	// About Mat
	//image Reading
	//Mat img = imread("test.png");
	//namedWindow("img", 0); // make window
	//imshow("img", img); //show
	//waitKey(0); // keep the window


	// video Reading
	VideoCapture cap(0); // open the default camera
	//VideoCapture capture("video dir"); // If you want to input other video 

	if (!cap.isOpened())
		return -1;

	namedWindow("video", 1); 

	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		imshow("video", frame);
		if (waitKey(30) >= 0) // If input the external key, destroy the video capture window
			break;
		// the camera iwll be deinitialized automatically in VideoCapture destructor
	}

	return 0;
}
