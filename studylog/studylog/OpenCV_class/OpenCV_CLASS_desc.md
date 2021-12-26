# OpenCV 주요 클래스

Source Code는 C++을 기반으로 작성했습니다. 

OpenCV에서 사용하는 클래스들을 알아봅시다. 좌표, 도형의 크기정보, 벡터 등 다양한 표현이 가능한 클래스 들이 존재합니다

[기본 자료형 클래스](https://www.notion.so/a5898256944b4198945889a014c2ddb1)

```cpp
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void PointOp();
void SizeOp();
void RectOp();
void RotatedRectOp();
void RangeOp();
void StringOp();

int main()
{
	PointOp();
	SizeOp();
	RectOp();
	RotatedRectOp();
	RangeOp();
	StringOp();

	return 0;
}

void PointOp()
{
	Point pt1;				// pt1 = (0, 0)
	pt1.x = 5; pt1.y = 10;	// pt1 = (5, 10)
	Point pt2(10, 30);		// pt2 = (10, 30)

	Point pt3 = pt1 + pt2;	// pt3 = [15, 40]
	Point pt4 = pt1 * 2;	// pt4 = [10, 20]
	int d1 = pt1.dot(pt2);	// d1 = 350
	bool b1 = (pt1 == pt2);	// b1 = false

	cout << "pt1: " << pt1 << endl;
	cout << "pt2: " << pt2 << endl;
}

void SizeOp()
{
	Size sz1, sz2(10, 20);			// sz1 = [0 x 0], sz2 = [10 x 20]
	sz1.width = 5; sz1.height = 10;	// sz1 = [5 x 10]

	Size sz3 = sz1 + sz2;	// sz3 = [15 x 30]
	Size sz4 = sz1 * 2;		// sz4 = [10 x 20]
	int area1 = sz4.area();	// area1 = 200

	cout << "sz3: " << sz3 << endl;
	cout << "sz4: " << sz4 << endl;
}

void RectOp()
{
	Rect rc1;					// rc1 = [0 x 0 from (0, 0)]
	Rect rc2(10, 10, 60, 40);	// rc2 = [60 x 40 from (10, 10)]

	Rect rc3 = rc1 + Size(50, 40);	// rc3 = [50 x 40 from (0, 0)]
	Rect rc4 = rc2 + Point(10, 10);	// rc4 = [60 x 40 from (20, 20)]

	Rect rc5 = rc3 & rc4;		// rc5 = [30 x 20 from (10, 10)]
	Rect rc6 = rc3 | rc4;		// rc6 = [80 x 60 from (0, 0)]

	cout << "rc5: " << rc5 << endl;
	cout << "rc6: " << rc6 << endl;
}

void RotatedRectOp()
{
	RotatedRect rr1(Point2f(40, 30), Size2f(40, 20), 30.f);

	Point2f pts[4];
	rr1.points(pts);

	Rect br = rr1.boundingRect();
}

void RangeOp()
{
	Range r1(0, 10);
}

void StringOp()
{
	String str1 = "Hello";
	String str2 = "world";
	String str3 = str1 + " " + str2;	// str3 = "Hello world"

	bool ret = (str2 == "WORLD");

	Mat imgs[3];
	for (int i = 0; i < 3; i++) {
		String filename = format("data%02d.bmp", i + 1);
		cout << filename << endl;
		imgs[i] = imread(filename);
	}
}
```

- Console 결과

    ![OpenCV%20%E1%84%8C%E1%85%AE%E1%84%8B%E1%85%AD%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A2%E1%84%89%E1%85%B3%207fa74ae796494924b6ab4c34c1b8b490/Untitled.png](OpenCV%20%E1%84%8C%E1%85%AE%E1%84%8B%E1%85%AD%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A2%E1%84%89%E1%85%B3%207fa74ae796494924b6ab4c34c1b8b490/Untitled.png)

주요 클래스들을 이용한 응용코드

```cpp
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
	Rect r1(100, 100, 320, 240), r2(200, 200, 320, 240); // 사각형의 위치와 크기정보를 나타내는 클래스
	Point pt1(100, 100); // 픽셀의 좌표를 표현하늨 클래스
	Size size(100, 100); // 사각형의 크기를 정의하는 클래스

	Rect r3 = r1 & r2;
	Rect r4 = r1 | r2;

	cout << "r1 :" << r1 << endl;
	cout << "r2 :" << r2 << endl;
	cout << "r3 :" << r3 << endl;
	cout << "r4 :" << r4 << endl;

	if (r1 != r2)
		cout << "r1 and r2 are not the same rectangle." << endl;

	// for drawing r1 ~ r4

	Mat img(600, 800, CV_8UC3); // x = 600, y = 800, 8bit, unsigned, 3 channels

	rectangle(img, r1, Scalar(255, 0, 0), 2);
	rectangle(img, r2, Scalar(0, 255, 0), 2);
	rectangle(img, r3, Scalar(0, 0, 255), 2);

	rectangle(img, r4, Scalar(0, 0, 0), 1);
	circle(img, pt1, 5, Scalar(255, 0, 255), 2);

	imshow("image", img);
	waitKey();

	return 0;
}
```

![OpenCV%20%E1%84%8C%E1%85%AE%E1%84%8B%E1%85%AD%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A2%E1%84%89%E1%85%B3%207fa74ae796494924b6ab4c34c1b8b490/Untitled%201.png](OpenCV%20%E1%84%8C%E1%85%AE%E1%84%8B%E1%85%AD%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A2%E1%84%89%E1%85%B3%207fa74ae796494924b6ab4c34c1b8b490/Untitled%201.png)

### Matx 클래스

고정된 작은 크기의 행렬을 위한 템플릿 클래스. 1 X 1 ~ 6 X 6의 작은 크기 행렬과 행렬연산 함수를 제공한다.

### Matx 클래스

Mat 클래스는 C++ API 에서 가장 중요한 클래스 중 하나로 1채널 또는 다채널의 실수, 복소수, 행렬, 영상 등의 수치 데이터를 표현하는 n 차원 행렬 클래스이다. Mat 클래스는 다양한 생성자와 메소드를 지원하게 된다. 구버전(3.0 미만)에서 사용되었던 CvMat, IplImage는 cvarrToMat 함수를 통해 호환하여 사용 할 수 있다.

### Vec 클래스

Mat 클래스에서 상속받은 클래스

짧은 수치 벡터를 위한 템플릿 클래스

기본적인 벡터 연산이 가능하고, 3차원 외적을 계산할 수 있고, []연산자에 의해 접근이 가능함.

고등학교 때 배운 벡터와 행렬이라고 생각하면 이해가 빠를 것이다

[vector 클래스 표](https://www.notion.so/02b1663a478b4d8bb6096cad1ad80eaa)

### Scalar 클래스

Vec 클래스에서 상속받은 4개의 요소를 갖는 템플릿 클래스이다. 계속 하위개념을 상속하는 형식이다.

Scalar X = Vec4f(1,2,3,4);

Scalar Y = Scalar(10,20,30); // = Scalar(10,20,30,0);

Scalar Z = Scalar(100,200,300);

이러한 형식으로 사용할 수 있다.

### InputArray 클래스

### OutputArray 클래스