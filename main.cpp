#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;
using namespace cv;

// Константы
const int COUNT_FRAME_SLEEP = 4;
//const int COUNT_FRAME_SLEEP = 0;
const int FRAME_LIMIT = 60;
const int CHEST_WIDTH = 7;
const int CHEST_HEIGTH = 7;
const double CHEST_SIZE = 5.0; // in mm
const double SENSOR_SIZE = 0.0055; // in mm
const Size CHEST_GRID = Size(CHEST_WIDTH, CHEST_HEIGTH);

// Values
Rect roi;
vector<Point2f> prevPointsGlobal;
Mat prevDescriptors;
Ptr<ORB> orb;
Ptr<BFMatcher> matcher;

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
    vector<uchar> pixels;
    pixels.assign(channel.datastart, channel.dataend);
    sort(pixels.begin(), pixels.end());

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
        vector<Mat> channels;
        cv::split(image, channels);
        v += computeMedian(channels[0]);
        v += computeMedian(channels[1]);
        v += computeMedian(channels[2]);
    }
    else {
        v = computeMedian(image);
    }

    lower = static_cast<int>(max(0.0, (1.0 - sigma) * v));
    upper = static_cast<int>(min(255.0, (1.0 + sigma) * v));

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
        vector<Mat> channels;
        cv::split(image, channels);
        v += computeMedian(channels[0]);
        v += computeMedian(channels[1]);
        v += computeMedian(channels[2]);
    }
    else {
        v = computeMedian(image);
    }

    int lower = static_cast<int>(max(0.0, (1.0 - sigma) * v));
    int upper = static_cast<int>(min(255.0, (1.0 + sigma) * v));

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
int New_Task_3(bool isWrite = true, bool isFirstVideo = true) {
    VideoCapture cap;
    if (isFirstVideo) cap = VideoCapture("C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/Calibration_01.mp4");
    else cap = VideoCapture("C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/Calibration_02.mp4");
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл." << endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    VideoWriter writer;

    if (isWrite) {
        if (isFirstVideo) writer = VideoWriter("lab_3_1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
        else writer = VideoWriter("lab_3_2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
        if (!writer.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для записи." << endl;
            return -1;
        }
    }

    ofstream logFile;
    if (isFirstVideo) logFile = ofstream("calibration_1.txt", ios::out | ios::binary);
    else logFile = ofstream("calibration_2.txt", ios::out | ios::binary);

    const string winName = "Calibration image";
    namedWindow(winName, WINDOW_NORMAL);
    resizeWindow(winName, min(1000, width), min(1000, height));

    Mat frame;
    int frameNum = 0;
    int frameFoundNum = 0;
    vector <vector<Point2f>> imagePoints;
    vector <vector<Point3f>> objectPoints;

    cout << "Подготовка данных калибровки" << endl;
    // Prepare object points (3D coordinates of chessboard corners in world space)
    vector<cv::Point3f> objectPointsInit;
    for (int i = 0; i < CHEST_HEIGTH; ++i) {
        for (int j = 0; j < CHEST_WIDTH; ++j) {
            objectPointsInit.push_back(cv::Point3f(j * CHEST_SIZE, i * CHEST_SIZE, 0.0f));
        }
    }
    bool isFirst = true;
    Size imgSize;
    cout << "Начало обработки кадров" << endl;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        if (isFirst) {
            imgSize = frame.size();
            isFirst = false;
        }
        frameNum++;

        if (COUNT_FRAME_SLEEP != 0 && frameNum % COUNT_FRAME_SLEEP != 0) continue;
        if (frameFoundNum >= FRAME_LIMIT) break;

        cv::Mat grayImage;
        cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);

        vector<Point2f> pointBuf;
        bool isFind = cv::findChessboardCorners(grayImage, CHEST_GRID, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (isFind)                // If done with success,
        {
            // improve the found corners' coordinate accuracy for chessboard
            cornerSubPix(grayImage, pointBuf, Size(11, 11),
                Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

            imagePoints.push_back(pointBuf);
            objectPoints.push_back(objectPointsInit);
            drawChessboardCorners(frame, CHEST_GRID, Mat(pointBuf), true);
            if (isWrite) writer.write(frame);
            frameFoundNum++;
        }
        else {
            cout << frameNum << " - Плохой кадр" << endl;
            cv::putText(frame, //target image
                "BAD FRAME", //text
                cv::Point(10, frame.rows / 2), //top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(118, 185, 0), //font color
                2);
        }

        cv::imshow(winName, frame);
        if (waitKey(1) == 27) break; // ESC
        grayImage.release();
        frame.release();
    }
    destroyAllWindows();
    cap.release();
    if (isWrite) writer.release();

    cout << "Обработка завершена." << endl;
    cout << "Калибровка по изображениям начата:" << endl;

    /// Calculate calibration parameters
    cv::Mat cameraMatrix, distCoeffs;
    vector<cv::Mat> rvecs, tvecs; // Rotation and translation vectors

    if (!imagePoints.empty()) {
        double rms = cv::calibrateCamera(objectPoints, imagePoints, imgSize,
            cameraMatrix, distCoeffs, rvecs, tvecs);

        cout << "Ошибка калибровки: " << rms << endl;
        logFile << u8"Ошибка калибровки: " << rms << "\n";

        cout << "Матрица камеры:\n" << cameraMatrix << endl;
        logFile << u8"Матрица камеры:\n" << cameraMatrix << "\n";

        cout << "Коэффициенты дисторсии:\n" << distCoeffs << endl;
        logFile << u8"Коэффициенты дисторсии:\n" << distCoeffs << "\n";

        // Сохранение в YAML (не зависит от кодировки текста)
        cv::FileStorage fs;
        if (isFirstVideo) fs = cv::FileStorage("camera_calibration_1.yml", cv::FileStorage::WRITE);
        else fs = cv::FileStorage("camera_calibration_2.yml", cv::FileStorage::WRITE);
        fs << "camera_matrix" << cameraMatrix;
        fs << "dist_coeffs" << distCoeffs;
        fs.release();

        // Извлечение параметров
        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);

        // Перевод в миллиметры
        double fx_mm = fx * SENSOR_SIZE;
        double fy_mm = fy * SENSOR_SIZE;
        double cx_mm = cx * SENSOR_SIZE;
        double cy_mm = cy * SENSOR_SIZE;

        // Фокусное расстояние
        cout << "Фокусное расстояние:\n";
        cout << "  fx = " << fx << " px  ->  " << fx_mm << " мм\n";
        cout << "  fy = " << fy << " px  ->  " << fy_mm << " мм\n";

        logFile << u8"Фокусное расстояние:\n";
        logFile << "  fx = " << fx << " px  ->  " << fx_mm << u8" мм\n";
        logFile << "  fy = " << fy << " px  ->  " << fy_mm << u8" мм\n";

        // Главная точка
        cout << "\nГлавная точка:\n";
        cout << "  cx = " << cx << " px  ->  " << cx_mm << " мм\n";
        cout << "  cy = " << cy << " px  ->  " << cy_mm << " мм\n";

        logFile << u8"\nГлавная точка:\n";
        logFile << "  cx = " << cx << " px  ->  " << cx_mm << u8" мм\n";
        logFile << "  cy = " << cy << " px  ->  " << cy_mm << u8" мм\n";

        // Смещение от центра
        double sensor_width_mm = imgSize.width * SENSOR_SIZE;
        double sensor_height_mm = imgSize.height * SENSOR_SIZE;
        double center_x_mm = sensor_width_mm / 2.0;
        double center_y_mm = sensor_height_mm / 2.0;

        double offset_x_mm = cx_mm - center_x_mm;
        double offset_y_mm = cy_mm - center_y_mm;

        cout << "\nСмещение главной точки от центра сенсора:\n";
        cout << "  dx = " << offset_x_mm << " мм\n";
        cout << "  dy = " << offset_y_mm << " мм\n";

        logFile << u8"\nСмещение главной точки от центра сенсора:\n";
        logFile << "  dx = " << offset_x_mm << u8" мм\n";
        logFile << "  dy = " << offset_y_mm << u8" мм\n";
    }
    else {
        cout << u8"Ошибка калибровки!!!!" << endl;
        logFile << u8"Ошибка калибровки!!!!\n";
    }

    cout << "Кадров обработано: " << frameNum << endl;
    logFile << u8"Кадров обработано: " << frameNum << "\n";
    cout << "Успешно найденых кадров: " << frameFoundNum << endl;
    logFile << u8"Успешно найденых кадров: " << frameFoundNum << "\n";
    cout << "Размер изображения: " << imgSize.width << "x" << imgSize.height << endl;
    logFile << u8"Размер изображения: " << imgSize.width << "x" << imgSize.height << "\n";

    logFile.close();
}
#pragma endregion

int main()
{
    setlocale(LC_ALL, "Russian");

    New_Task_3(true, true);
    return 0;
}

void Logs()
{
    ofstream logs;
    logs.open("report.txt");
    cout << logs.is_open() << endl;
    logs << "////////////////////////////////////////////////////////////////" << "\n";
    logs.close();
}