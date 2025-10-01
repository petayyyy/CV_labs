#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;
using namespace cv;

// Константы
const int MAX_FEATURES = 1000;
const double MAX_DISTANCE_POINTS = 50;

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
bool createVideo(VideoCapture cap, VideoWriter& writer, string nameF, Mat img) {
    // Получение параметров видео
    int frame_width = img.rows;
    int frame_height = img.cols;
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

#pragma region Function
Rect CalAreaInterest(Mat binImg) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    // Horasion
    int maxContidG;
    float sumArG = 0;
    // Car
    int maxContidC;
    float sumArC = 0;
    findContours(binImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Moments M = cv::moments(contours[i]);
        double newArea = cv::contourArea(contours[i]);
        if (M.m01 / M.m00 < binImg.rows / 2) {
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
    //Point center_horasionR = (horasionR.br() + horasionR.tl()) * 0.5;
    Point center_horasionR = horasionR.br();
    auto carR = boundingRect(contours.at(maxContidC));
    return  Rect (0, center_horasionR.y, binImg.cols, (carR.y - center_horasionR.y));
}
#pragma endregion

#pragma region Utils
Rect computeROI(const Mat& frame) {
    return Rect(int(1.0f/4.0f * (float)frame.cols), int(float(frame.rows) * 1.0f / 3.0f), int(3.0f / 5.0f * (float)frame.cols), int(float(frame.rows) * 2.0f / 4.0f));
}
double distanceP(Point2f a, Point2f b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
#pragma endregion

#pragma region Tasks
int New_Task_2(bool isWrite = true) {
    VideoCapture cap("C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/Movement 01.mp4");
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл." << endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    VideoWriter writer;

    if (isWrite) {
        writer = VideoWriter("lab_2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
        if (!writer.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для записи." << endl;
            return -1;
        }
    }

    ofstream logFile("frame_stats.csv");
    logFile << "frame;detected;matched;total_ms;detect_ms;match_ms\n";

    Ptr<ORB> orb = ORB::create(MAX_FEATURES);
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING, true);

    vector<Point2f> prevPointsGlobal;
    Mat prevDescriptors;

    const string winName = "ORB Feature Tracks (2-frame)";
    namedWindow(winName, WINDOW_NORMAL);
    resizeWindow(winName, min(1280, width), min(720, height));

    Mat frame;
    Rect roi;
    int frameNum = 0;
    bool isFirst = true;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        if (isFirst) {
            roi = computeROI(frame);
            isFirst = false;
        }
        Mat roiImg = frame(roi);

        rectangle(frame, Point(roi.x, roi.y), Point(roi.x + roi.width, roi.y + roi.height), Scalar(255, 255, 0), 5, LINE_8, 0);

        auto startDetect = chrono::high_resolution_clock::now();
        vector<KeyPoint> currKeypoints;
        Mat currDescriptors;
        orb->detectAndCompute(roiImg, Mat(), currKeypoints, currDescriptors);
        auto stopDetect = chrono::high_resolution_clock::now();
        double detectMs = chrono::duration_cast<chrono::microseconds>(stopDetect - startDetect).count() / 1000.0;
        int detected = (int)currKeypoints.size();

        vector<DMatch> matches;
        double matchMs = 0.0;
        int matched = 0;

        if (!prevDescriptors.empty() && !currDescriptors.empty()) {
            auto startMatch = chrono::high_resolution_clock::now();
            matcher->match(prevDescriptors, currDescriptors, matches);
            auto stopMatch = chrono::high_resolution_clock::now();
            matchMs = chrono::duration_cast<chrono::microseconds>(stopMatch - startMatch).count() / 1000.0;
        }

        vector<Point2f> currPointsGlobal;
        for (const auto& kp : currKeypoints) {
            currPointsGlobal.push_back(Point2f(kp.pt.x + roi.x, kp.pt.y + roi.y));
        }

        for (const auto& m : matches) {
            int prevIdx = m.queryIdx;
            int currIdx = m.trainIdx;
            if (prevIdx < (int)prevPointsGlobal.size() && currIdx < (int)currPointsGlobal.size()) {
                if (distanceP(prevPointsGlobal[prevIdx], currPointsGlobal[currIdx]) < MAX_DISTANCE_POINTS) {
                    line(frame, prevPointsGlobal[prevIdx], currPointsGlobal[currIdx], Scalar(255, 0, 0), 1);
                    matched++;
                }
                circle(frame, currPointsGlobal[currIdx], 3, Scalar(0, 0, 255), -1);
            }
        }

        vector<bool> usedCurr(currKeypoints.size(), false);
        for (const auto& m : matches) {
            if (m.trainIdx < (int)usedCurr.size())
                usedCurr[m.trainIdx] = true;
        }
        for (size_t i = 0; i < currKeypoints.size(); ++i) {
            if (!usedCurr[i]) {
                circle(frame, currPointsGlobal[i], 3, Scalar(0, 0, 255), -1);
            }
        }

        double totalMs = detectMs + matchMs;
        logFile << frameNum << ";"
            << detected << ";"
            << matched << ";"
            << totalMs << ";"
            << detectMs << ";"
            << matchMs << "\n";

        currPointsGlobal.swap(prevPointsGlobal);
        currDescriptors.copyTo(prevDescriptors);

        if (isWrite) writer.write(frame);
        imshow(winName, frame);
        imshow("ROI", frame(roi));
        if (waitKey(1) == 27) break; // ESC

        frameNum++;
    }

    cap.release();
    if (isWrite) writer.release();
    logFile.close();
    destroyAllWindows();

    cout << "Обработка завершена." << endl;
    cout << "Видео: lab_2.avi" << endl;
    cout << "Лог: frame_stats.csv" << endl;
    cout << "Кадров обработано: " << frameNum << endl;
}
#pragma endregion

int main()
{
    setlocale(LC_ALL, "Russian");

    New_Task_2(true);
    return 0;
}

void Logs()
{
    std::ofstream logs;
    logs.open("report.txt");
    cout << logs.is_open() << endl;
    logs << "////////////////////////////////////////////////////////////////" << "\n";
    logs.close();
}