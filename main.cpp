#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

#pragma region File
bool createVideo(VideoCapture cap, VideoWriter& writer, string nameF, bool isResize = false) {
    // Получение параметров видео
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    if (isResize) {
        frame_width /= 4;
        frame_height /= 4;
    }
    double fps = cap.get(CAP_PROP_FPS);

    // Создание объекта для записи видео
    writer = VideoWriter(nameF + ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
        Size(frame_width, frame_height));

    if (!writer.isOpened()) {
        cerr << "Не удается создать запись" << endl;
        return false;
    }
    return true;
}

#pragma endregion

#pragma region Image process
double computeMedian(Mat channel) {
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
/// <summary>
/// Auto canny edge detection - https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
/// </summary>
/// <param name="image"></param>
/// <param name="sigma"></param>
/// <returns></returns>
Mat autoCanny(const Mat& image, int& lower, int& upper, double sigma = 0.33) {
    double v = 0.0;
    if (image.channels() > 1) {
        std::vector<Mat> channels;
        cv::split(image, channels);
        v += computeMedian(channels[0]);
        v += computeMedian(channels[1]);
        v += computeMedian(channels[2]);
    }
    else {
        v = computeMedian(image);
    }

    lower = static_cast<int>(std::max(0.0, (1.0 - sigma) * v));
    upper = static_cast<int>(std::min(255.0, (1.0 + sigma) * v));

    cv::Mat edged;
    cv::Canny(image, edged, lower, upper);

    return edged;
}
/// <summary>
/// Auto canny edge detection - https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
/// </summary>
/// <param name="image"></param>
/// <param name="sigma"></param>
/// <returns></returns>
Mat autoCanny(const Mat& image, double sigma = 0.33) {
    double v = 0.0;
    if (image.channels() > 1) {
        std::vector<Mat> channels;
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
void dilateImg(Mat& inpImg, bool isErode = true) {
    int morph_size = 10;
    Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, Size(2 * morph_size + 1,
            2 * morph_size + 1),
        Point(morph_size, morph_size));
    cv::dilate(inpImg, inpImg, kernel, Point(-1, -1), 1);
    if (isErode) cv::erode(inpImg, inpImg, kernel, Point(-1, -1), 1);
    kernel.release();
}
#pragma endregion

#pragma region Tasks
int Task_2(std::ofstream& logs, bool isLine = false, bool isResize = false, bool isWrite = true) {
    namedWindow("Input image", WINDOW_AUTOSIZE);

    VideoCapture cap; VideoWriter writer;
    cap.open("C:/Users/ilyah/Desktop/IS_RTK/Video/20180305_1337_Cam_1_07_00 Part.mp4");

    if (!cap.isOpened()) {
        cerr << "Не удается открыть видео файл" << endl;
        return -1;
    }
    if (isWrite && !createVideo(cap, writer, !isResize ? "lab2" : "lab2_resize", true)) {
        return -1;
    }

    Mat frame;
    int frame_count = 0;
    double total_time = 0;
    double total_upC = 0;
    double total_lowC = 0;
    double total_timeR = 0;

    for (;;)
    {
        cap >> frame;
        if (frame.empty())
            break;
        if (isResize) 
        {
            auto startR = chrono::high_resolution_clock::now();
            cv::resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));
            // stop timer
            auto stopR = chrono::high_resolution_clock::now();
            auto durationR = chrono::duration_cast<chrono::milliseconds>(stopR - startR);
            total_timeR += durationR.count();
        }

        // start timer
        auto start = chrono::high_resolution_clock::now();
        Mat imgProc = frame.clone();
        cv::cvtColor(imgProc, imgProc, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(imgProc, imgProc, Size(7, 7), 0, 0);
        int low, up = 0;
        Mat edges = autoCanny(imgProc, low, up);
        total_lowC += low;
        total_upC += up;
        dilateImg(edges);

        // stop timer
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        total_time += duration.count();

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        // Horasion
        int maxContidG;
        float sumArG = 0;

        findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        for (size_t i = 0; i < contours.size(); i++)
        {
            drawContours(frame, contours, (int)i, Scalar(255, 0, 0), 2, LINE_8, hierarchy, 0);
            if (isLine) {
                cv::Moments M = cv::moments(contours[i]);
                double newArea = cv::contourArea(contours[i]);
                if (M.m01 / M.m00 < frame.rows / 2) {
                    if (newArea > sumArG) {
                        sumArG = newArea;
                        maxContidG = i;
                    }
                }
            }
        }
        if (isLine) {
            Vec4f line;
            fitLine(contours.at(maxContidG), line, DIST_L1, 0, 0.01, 0.01);
            float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
            Point2f pt1(x0 - vx * 1000, y0 - vy * 1000); // Start point of line
            Point2f pt2(x0 + vx * 1000, y0 + vy * 1000); // End point of line
            cv::line(frame, (Point)pt1, (Point)pt2, Scalar(0, 0, 255), 2);
        }
        // Запись кадра в видеофайл
        if (isWrite) writer.write(frame);

        //imshow("canny", edges);
        imshow("Input image", frame);
        frame_count++;

        if (cv::waitKey(27) >= 0)
            break;
        frame.release();
        edges.release();
        imgProc.release();
    }

    setlocale(LC_ALL, "Russian");
    // Вывод статистики
    cout << "Общее количество кадров в файле: " << frame_count << endl;
    cout << "Среднее время обработки кадра : " << total_time / frame_count << " ms" << endl;
    if (isResize) cout << "Среднее время уменьшения размера кадра : " << total_timeR / frame_count << " ms" << endl;
    cout << "Среднее значение threshold1 детектора Cannny : " << total_lowC / frame_count << endl;
    cout << "Среднее значение threshold2 детектора Cannny : " << total_upC / frame_count << endl;
    
    // Report
    logs << "Общее количество кадров в файле: " << frame_count << "\n";
    logs << "Среднее время обработки кадра : " << total_time / frame_count << " ms" << "\n";
    if (isResize) logs << "Среднее время уменьшения размера кадра : " << total_timeR / frame_count << " ms" << "\n";
    logs << "Среднее значение threshold1 детектора Cannny : " << total_lowC / frame_count << "\n";
    logs << "Среднее значение threshold2 детектора Cannny : " << total_upC / frame_count << "\n";

    // clear
    cap.release();
    writer.release();
    destroyAllWindows();
}
int Task_3(bool isWrite = true) {
    namedWindow("In image", WINDOW_AUTOSIZE);

    VideoCapture cap; VideoWriter writer;
    cap.open("C:/Users/ilyah/Desktop/IS_RTK/Video/20180305_1337_Cam_1_07_00 Part.mp4");

    if (!cap.isOpened()) {
        cerr << "Не удается открыть видео файл" << endl;
        return -1;
    }
    if (isWrite && !createVideo(cap, writer, "lab3")) {
        return -1;
    }

    Mat frame;
    int frame_count = 0;
    double total_time = 0;

    for (;;)
    {
        cap >> frame;
        if (frame.empty())
            break;

        Mat imgProc = frame.clone();
        cv::cvtColor(imgProc, imgProc, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(imgProc, imgProc, Size(7, 7), 0, 0);
        Mat edges = autoCanny(imgProc);
        dilateImg(edges, false);

        // calculate rect 
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
            //drawContours(frame, contours, (int)i, Scalar(255, 0, 0), 2, LINE_8, hierarchy, 0);
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

        auto horasionR = boundingRect(contours.at(maxContidG));
        Point center_horasionR = (horasionR.br() + horasionR.tl()) * 0.5;
        auto carR = boundingRect(contours.at(maxContidC));

        Rect outRect(0, center_horasionR.y, frame.cols, (carR.y - center_horasionR.y));
        cv::rectangle(frame, outRect, Scalar(0, 0, 255), 3);

        // Запись кадра в видеофайл
        if (isWrite && writer.isOpened()) writer.write(frame);

        //imshow("canny", edges);
        imshow("In image", frame);
        if (waitKey(1) == 27) break; // ESC для выхода

        frame_count++;

        if (cv::waitKey(27) >= 0)
            break;
        frame.release();
        edges.release();
        imgProc.release();
    }

    setlocale(LC_ALL, "Russian");
    // Вывод статистики
    cout << "Общее количество кадров в файле: " << frame_count << endl;
    cout << "Среднее время обработки кадра : " << total_time / frame_count << " ms" << endl;
    //! запись в файл

    // clear
    cap.release();
    writer.release();
    destroyAllWindows();
}
#pragma endregion

int main()
{
    std::ofstream logs;
    logs.open("report.txt");
    cout << logs.is_open() << endl;

    // size in == out
    Task_2(logs, true, false, true);
    cout << "////////////////////////////////////////////////////////////////" << endl;
    logs << "////////////////////////////////////////////////////////////////" << "\n";
    // size in / 4 == out
    Task_2(logs, true, true, true);

    logs.close();
    return 0;
}