#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace std;
using namespace cv;

bool createVideo(VideoCapture cap, VideoWriter& writer, string nameF) {
    // Получение параметров видео
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    // Создание объекта для записи видео
    writer = VideoWriter(nameF + ".mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
        Size(frame_width, frame_height), false);

    if (!writer.isOpened()) {
        cerr << "Не удается создать запись" << endl;
        return false;
    }
    return true;
}
double computeMedian(cv::Mat channel) {
    double median = 0.0;
    std::vector<uchar> pixels;
    pixels.assign(channel.datastart, channel.dataend);
    std::sort(pixels.begin(), pixels.end());

    if (pixels.size() % 2 == 0) {
        median = (pixels[pixels.size() / 2 - 1] + pixels[pixels.size() / 2]) / 2.0;
    }
    else {
        median = pixels[pixels.size() / 2];
    }

    return median;
}

cv::Mat autoCanny(const cv::Mat& image, double sigma = 0.33) {
    double v = 0.0;
    if (image.channels() > 1) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        v += computeMedian(channels[0]);
        v += computeMedian(channels[1]);
        v += computeMedian(channels[2]);
    }
    else {
        v = computeMedian(image);
    }

    int lower = static_cast<int>(std::max(0.0, (1.0 - sigma) * v));
    int upper = static_cast<int>(std::min(255.0, (1.0 + sigma) * v));

    cv::Mat edged;
    cv::Canny(image, edged, lower, upper);

    return edged;
}

int main()
{
	namedWindow("Example3", WINDOW_AUTOSIZE);

	VideoCapture cap; VideoWriter writer;
	cap.open("C:/Users/ilyah/Desktop/IS_RTK/Video/20180305_1337_Cam_1_07_00 Part.mp4");

    if (!cap.isOpened()) {
        cerr << "Не удается открыть видео файл" << endl;
        return -1;
    }
    if (!createVideo(cap, writer, "lab2")) {
        return -1;
    }   
	
    Mat frame, edges;
    int frame_count = 0;
    double total_time = 0;

	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break; // Видеофайл завершился - выход

        auto start = high_resolution_clock::now();
        Mat imgProc = frame.clone();
        cvtColor(imgProc, imgProc, COLOR_BGR2GRAY);
        GaussianBlur(imgProc, imgProc, Size(7, 7), 0, 0);
        Mat edges = autoCanny(imgProc);
        int morph_size = 10;
        Mat kernel = getStructuringElement(
            MORPH_RECT, Size(2 * morph_size + 1,
                2 * morph_size + 1),
            Point(morph_size, morph_size));
        dilate(edges, edges, kernel, Point(-1, -1), 1);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        total_time += duration.count();

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        // Horasion
        int maxContidG;
        float sumArG = 0;
        // Car
        int maxContidC;
        float sumArC = 0;
        findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for (size_t i = 0; i < contours.size(); i++)
        {
            drawContours(frame, contours, (int)i, Scalar(255, 0, 0), 2, LINE_8, hierarchy, 0);
            cv::Moments M = cv::moments(contours[i]);
            double newArea = cv::contourArea(contours[i]);
            if (M.m01 / M.m00 < frame.rows / 2) {
                if (newArea > sumArG) {
                    sumArG = newArea;
                    maxContidG = i;
                }
            }
            else {
                if (newArea > sumArC) {
                    sumArC = newArea;
                    maxContidC = i;
                }
            }
        }
        Vec4f line;
        //fitLine(contours.at(maxContid), line, DIST_L2, 0, 0.01, 0.01);
        fitLine(contours.at(maxContidG), line, DIST_L1, 0, 0.01, 0.01);
        float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
        Point2f pt1(x0 - vx * 1000, y0 - vy * 1000); // Start point of line
        Point2f pt2(x0 + vx * 1000, y0 + vy * 1000); // End point of line
        cv::line(frame, (Point)pt1, (Point)pt2, Scalar(0, 0, 255), 2);

        auto horasionR = boundingRect(contours.at(maxContidG));
        Point center_horasionR = (horasionR.br() + horasionR.tl()) * 0.5;
        auto carR = boundingRect(contours.at(maxContidC));

        Rect outRect(0, center_horasionR.y, frame.cols, (carR.y - center_horasionR.y));
        cv::rectangle(frame, outRect, Scalar(0, 0, 255), 3);
        // Запись кадра в видеофайл
        //writer.write(edgesD);

        //imshow("canny", edges);
        imshow("Example3", frame);
        if (waitKey(1) == 27) break; // ESC для выхода

        frame_count++;

		if (cv::waitKey(27) >= 0)
			break;
        frame.release();
        edges.release();
        kernel.release();
	}

    // Вывод статистики
    cout << "Total frames processed: " << frame_count << endl;
    cout << "Average processing time per frame: " << total_time / frame_count << " ms" << endl;

    // Освобождение ресурсов
    cap.release();
    writer.release();
    destroyAllWindows();

    return 0;
}