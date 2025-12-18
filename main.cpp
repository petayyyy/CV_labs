#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/objdetect/detection_based_tracker.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;
using namespace std::chrono;

string GLOBAL_RESULTS_PATH = "C:/Users/ilyah/Desktop/MGY/IS_RTK/result_4/";

#pragma region Function
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
Mat adaptiveCanny(const Mat& gray, double sigma1 = 0.1, double sigma2 = 0.3) {
    // Вычисляем пороги для двух разных сигм
    double v = computeMedian(gray);

    int lower1 = static_cast<int>(max(0.0, (1.0 - sigma1) * v));
    int upper1 = static_cast<int>(min(255.0, (1.0 + sigma1) * v));

    int lower2 = static_cast<int>(max(0.0, (1.0 - sigma2) * v));
    int upper2 = static_cast<int>(min(255.0, (1.0 + sigma2) * v));

    Mat edges1, edges2;
    Canny(gray, edges1, lower1, upper1);
    Canny(gray, edges2, lower2, upper2);

    // Объединяем результаты
    Mat edges = edges1 | edges2;

    return edges;
}
#pragma endregion

#pragma region Tasks
void New_Task_4(bool isWrite = true, bool isView = false, bool isFirst = true, bool isContours = true) {
    // Открываем видеофайл
    VideoCapture cap;
    string videoPath;
    if (isFirst) videoPath = "C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/20191119_1241_Cam_1_16_00.mp4";
    else videoPath = "C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/20180305_1337_Cam_1_07_00 Part.mp4";
    cap.open(videoPath);
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл: " << videoPath << endl;
        return;
    }

    // Получаем параметры видео
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    cout << "Видео: " << videoPath << endl;
    cout << "Размер: " << width << "x" << height << ", FPS: " << fps << endl;

    // Создаем объекты для записи видео
    VideoWriter writer_result;
    VideoWriter writer_result2;
    string baseName = "lab4" + string(isFirst ? "_1" : "_2") + string(isContours ? "_contours" : "_pixels");

    if (isWrite) {
        writer_result.open(GLOBAL_RESULTS_PATH + baseName + "_result.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            Size(640, 480));
        if (!writer_result.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для векторного представления." << endl;
            return;
        }
        writer_result2.open(GLOBAL_RESULTS_PATH + baseName + "_result2.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            Size(640, 480));
        if (!writer_result2.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для векторного представления." << endl;
            return;
        }
    }

    // Создаем окно для отображения
    const string winName = "Detect_object";
    const string winName2 = "Mask_object";
    if (isView) {
        namedWindow(winName, WINDOW_NORMAL);
        resizeWindow(winName, min(1200, width), min(1000, height));
        namedWindow(winName2, WINDOW_NORMAL);
        resizeWindow(winName2, min(1200, width), min(1000, height));
    }

    // Переменные для обработки
    int frameNum = 0;
    Mat frame, frameSmall, segmented, gray, edges, contoursOverlay;

    vector<double> processingTimes;

    // Создаем файл статистики
    ofstream statsFile(GLOBAL_RESULTS_PATH + baseName + "_stats.txt");
    if (!statsFile.is_open()) {
        cerr << "Ошибка: не удалось создать файл статистики" << endl;
        return;
    }

    // Записываем заголовок в файл статистики
    statsFile << "Frame ProcessingTime(ms)" << endl;

    cout << "Начало обработки кадров..." << endl;

    // Основной цикл обработки видео
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::resize(frame, frameSmall, Size(640, 480), 0, 0, cv::INTER_LINEAR);
        auto start = chrono::high_resolution_clock::now();

        // 1. Применить MeanShift
        pyrMeanShiftFiltering(frameSmall, segmented, 8, 12, 1);

        // 3. Преобразовать в оттенки серого и применить Canny
        cvtColor(segmented, gray, COLOR_BGR2GRAY);
        edges = adaptiveCanny(gray);
        //Canny(gray, edges, 50, 150);

        // 4. Наложить контуры на сегментированное изображение
        contoursOverlay = segmented.clone(); // Копируем сегментированное изображение

        if (isContours) {
            // Вариант 1: Использовать findContours/drawContours
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(contoursOverlay, contours, -1, Scalar(0, 255, 0), 1, LINE_8); // Зеленые контуры
        }
        else {
            // Вариант 2: Попиксельная обработка

            for (int y = 0; y < edges.rows; ++y) {
                for (int x = 0; x < edges.cols; ++x) {
                    if (edges.at<uchar>(y, x) != 0) { // Если пиксель является частью контура
                        contoursOverlay.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
                    }
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0; // в мс
        processingTimes.push_back(duration);
        statsFile << frameNum << " " << duration << endl;
        
        // Записываем кадры в видеофайлы
        if (isWrite) {
            // 2. Записать сегментированное изображение
            writer_result.write(segmented);

            // 5. Записать изображение с наложенными контурами
            writer_result2.write(contoursOverlay);
        }

        // Отображаем результат
        if (isView) {
            imshow(winName, segmented);
            imshow(winName2, contoursOverlay);
        }
        

        frameNum++;
        // Выход по нажатию ESC
        if (isView && waitKey(1) == 27) {
            cout << "Обработка прервана пользователем." << endl;
            break;
        }

        // Выводим прогресс каждые 10 кадров
        if (frameNum % 10 == 0) {
            cout << "Обработано кадров: " << frameNum << endl;
        }
    }

    // Закрываем все ресурсы
    destroyAllWindows();
    cap.release();
    if (isWrite) {
        writer_result.release();
        writer_result2.release();
    }
    statsFile.close();
    cout << "Обработка завершена. Обработано кадров: " << frameNum << endl;
}
#pragma endregion

int main()
{
    setlocale(LC_ALL, "Russian");

    bool isView = false;      // Показывать окно с результатом
    bool isWrite = true;     // Сохранять результаты

    New_Task_4(isWrite, isView, false, true);
    New_Task_4(isWrite, isView, true, true);

    New_Task_4(isWrite, isView, false, false);
    New_Task_4(isWrite, isView, true, false);
    return 0;
}