#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	cv::namedWindow("Example3", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture cap;
	bool ret = cap.open("C:/Users/ilyah/Downloads/uu.mp4");
	if (ret) {
		cout << "Video finding!" << endl;
	}
	cv::Mat frame;
	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break; // Видеофайл завершился - выход

		cv::imshow("Example3", frame);
		if (cv::waitKey(27) >= 0)
			break;
	}

	return 0;
}
