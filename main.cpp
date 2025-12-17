#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/videoio/videoio.hpp"

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

// Структура для хранения параметров алгоритма
struct FarnebackParams {
    double pyrScale; // отношение масштабов уровней пирамиды (< 1.0) 
    int levels; // число уровней пирамиды
    int winsize; // размер окна на этапе предварительного сглаживания
    int iterations; // число итераций на каждом уровне пирамиды
    int polyN; // размер окрестности для аппроксимации полиномом
    double polySigma; // стандартное отклонение ядра Гаусса для вычисления
    int flags; // флаги режимов вычислений
    string nameParams;
};

string GLOBAL_RESULTS_PATH = "C:/Users/ilyah/Desktop/MGY/IS_RTK/result_5/";

#pragma region Function
// Функция для визуализации оптического потока в цвете (направление - цвет, величина - яркость)
Mat drawOpticalFlowColor(const Mat& flow) {
    // Разделяем поток на компоненты X и Y
    Mat flow_parts[2];
    split(flow, flow_parts);

    // Вычисляем величину и угол (в градусах) для каждого вектора
    Mat magnitude, angle;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

    // Нормализуем величину в диапазон [0, 1]
    Mat magnitude_normalized;
    normalize(magnitude, magnitude_normalized, 0.0f, 1.0f, NORM_MINMAX);

    // Преобразуем угол для использования в HSV (0-179)
    angle *= ((1.0f / 360.0f) * (180.0f / 255.0f));

    // Создаем HSV изображение
    Mat _hsv[3], hsv, hsv8, bgr;

    // Канал H (тон) - направление движения
    _hsv[0] = angle;

    // Канал S (насыщенность) - максимальная насыщенность (белый цвет)
    _hsv[1] = Mat::ones(angle.size(), CV_32F);

    // Канал V (значение, яркость) - величина смещения
    _hsv[2] = magnitude_normalized;

    // Объединяем каналы в одно HSV изображение
    merge(_hsv, 3, hsv);

    // Конвертируем из 32-битного float в 8-битный unsigned char
    hsv.convertTo(hsv8, CV_8U, 255.0);

    // Конвертируем HSV в BGR для отображения
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);

    return bgr;
}

// Функция для визуализации оптического потока в виде векторов
cv::Mat drawOpticalFlowVectors(const cv::Mat& flow, int gridStep = 20) {
    // Проверка входных данных
    CV_Assert(flow.type() == CV_32FC2);

    // Создаем черное изображение того же размера
    cv::Mat flowImage = cv::Mat::zeros(flow.size(), CV_8UC1);

    // Находим максимальную длину вектора для масштабирования
    float maxMagnitude = 0.0f;
    for (int y = 0; y < flow.rows; y += gridStep) {
        for (int x = 0; x < flow.cols; x += gridStep) {
            cv::Vec2f flowVec = flow.at<cv::Vec2f>(y, x);
            float magnitude = std::sqrt(flowVec[0] * flowVec[0] + flowVec[1] * flowVec[1]);
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
            }
        }
    }

    // Увеличиваем масштаб для лучшей видимости
    float scaleFactor = 2.0f; // Коэффициент увеличения масштаба
    float scale = (maxMagnitude > 0) ?
        (static_cast<float>(gridStep) / maxMagnitude) * scaleFactor :
        scaleFactor;

    // Рисуем векторы с заданным шагом
    for (int y = gridStep / 2; y < flow.rows; y += gridStep) {
        for (int x = gridStep / 2; x < flow.cols; x += gridStep) {
            cv::Vec2f flowVec = flow.at<cv::Vec2f>(y, x);

            // Рисуем ВСЕ векторы, даже если они маленькие
            // (увеличим минимальную длину для лучшей видимости)
            float magnitude = std::sqrt(flowVec[0] * flowVec[0] + flowVec[1] * flowVec[1]);

            // Минимальная длина отображаемого вектора (в пикселях)
            float minDisplayLength = 2.0f;

            // Если вектор слишком короткий, рисуем его с минимальной длиной
            if (magnitude * scale < minDisplayLength) {
                if (magnitude > 0) {
                    // Нормализуем и умножаем на минимальную длину
                    flowVec[0] = (flowVec[0] / magnitude) * (minDisplayLength / scale);
                    flowVec[1] = (flowVec[1] / magnitude) * (minDisplayLength / scale);
                }
                else {
                    // Если вектор нулевой, рисуем маленькую точку
                    flowVec[0] = 0;
                    flowVec[1] = 0;
                }
            }

            // Вычисляем конечную точку вектора
            int endX = static_cast<int>(x + flowVec[0] * scale);
            int endY = static_cast<int>(y + flowVec[1] * scale);

            // Ограничиваем точку в пределах изображения
            endX = std::max(0, std::min(endX, flow.cols - 1));
            endY = std::max(0, std::min(endY, flow.rows - 1));

            // Рисуем начальную точку (кружок)
            cv::circle(flowImage,
                cv::Point(x, y),
                1,  // Радиус 1 пиксель
                cv::Scalar(255),  // Белый цвет
                -1,  // Заполненный круг
                cv::LINE_AA);

            // Если вектор не нулевой, рисуем линию
            if (flowVec[0] != 0 || flowVec[1] != 0) {
                // Рисуем отрезок (вектор) белым цветом
                cv::line(flowImage,
                    cv::Point(x, y),
                    cv::Point(endX, endY),
                    cv::Scalar(255),  // Белый цвет
                    1,                // Толщина линии
                    cv::LINE_AA);     // Сглаживание

                // Рисуем окружность в конце вектора
                cv::circle(flowImage,
                    cv::Point(endX, endY),
                    2,  // Радиус 2 пикселя (диаметр 4)
                    cv::Scalar(255),  // Белый цвет
                    -1,               // Заполненный круг
                    cv::LINE_AA);     // Сглаживание
            }
        }
    }

    return flowImage;
}

// Функция для вычисления статистики по области интереса
void computeFlowStats(const Mat& flow,
    double& minMag, double& maxMag, double& meanMag,
    int& devBelow, int& devAbove, double thresholdPercent = 0.2) {
    minMag = DBL_MAX;
    maxMag = 0;
    meanMag = 0;
    devBelow = 0;
    devAbove = 0;

    if (flow.empty()) {
        return;
    }

    // Вычисляем статистику по ROI
    int count = 0;

    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            double mag = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);

            if (mag < minMag) minMag = mag;
            if (mag > maxMag) maxMag = mag;
            meanMag += mag;
            count++;
        }
    }

    if (count > 0) {
        meanMag /= count;
    }

    // Вычисляем отклонения от среднего
    double range = maxMag - minMag;
    double threshold = thresholdPercent * range;

    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            double mag = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);

            if (mag < meanMag - threshold) {
                devBelow++;
            }
            else if (mag > meanMag + threshold) {
                devAbove++;
            }
        }
    }
}

// Функция для выделения движущихся объектов в ROI
Mat highlightMovingObjects(const Mat& frame, const Mat& flow, const Rect& roi,
    double minMag, double maxMag, double meanMag,
    double thresholdPercent = 0.2) {
    Mat result = frame.clone();

    // Проверяем, что ROI находится в пределах изображения и flow не пустой
    Rect validRoi = roi & Rect(0, 0, frame.cols, frame.rows);
    if (validRoi.area() == 0 || flow.empty()) {
        rectangle(result, roi, Scalar(0, 255, 0), 2);
        return result;
    }

    // Рисуем прямоугольник ROI
    rectangle(result, validRoi, Scalar(255, 190, 190), 2);

    // Выделяем пиксели с отклонениями
    double range = maxMag - minMag;
    double threshold = thresholdPercent * range;

    // Создаем маску для движущихся объектов (только в ROI)
    Mat movingMask = Mat::zeros(validRoi.size(), CV_8UC1);

    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            // Координаты в оригинальном изображении
            int origX = validRoi.x + x;
            int origY = validRoi.y + y;

            if (origX >= frame.cols || origY >= frame.rows) continue;

            const Point2f& fxy = flow.at<Point2f>(y, x);
            double mag = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);

            // Если величина отклоняется от среднего более чем на порог
            if (abs(mag - meanMag) > threshold) {
                movingMask.at<uchar>(y, x) = 255;
            }
        }
    }

    // Применяем морфологические операции для устранения шума
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(movingMask, movingMask, MORPH_CLOSE, kernel);

    // Находим контуры движущихся объектов
    vector<vector<Point>> contours;
    findContours(movingMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Рисуем контуры и bounding boxes на оригинальном изображении
    for (size_t i = 0; i < contours.size(); i++) {
        // Смещаем контур к координатам оригинального изображения
        vector<Point> offsetContour;
        for (const auto& p : contours[i]) {
            offsetContour.push_back(Point(p.x + validRoi.x, p.y + validRoi.y));
        }
        drawContours(result, vector<vector<Point>>{offsetContour}, 0, Scalar(0, 0, 255), 2);

        if (contourArea(contours[i]) > 400) { // Игнорируем маленькие контуры
            // Рисуем bounding box в координатах оригинального изображения
            Rect bbox = boundingRect(contours[i]);
            bbox.x += validRoi.x;
            bbox.y += validRoi.y;
            rectangle(result, bbox, Scalar(255, 0, 0), 3);
        }
    }
    return result;
    //return movingMask;
}
#pragma endregion

#pragma region Utils
// Функция для вычисления области интереса (передняя часть автомобиля)
Rect computeROI(const Mat& frame) {
    int x = frame.cols / 6;
    int y = frame.rows * 1 / 3;
    int width = frame.cols * 2 / 3;
    int height = frame.rows / 3;
    return Rect(x, y, width, height);
}
Size computeROISize(Size in_size) {
    int width = in_size.width * 2 / 3;
    int height = in_size.height / 3;
    return Size(width, height);
}

// Функция для вычисления расстояния между двумя точками
double distanceP(Point2f a, Point2f b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}
#pragma endregion

#pragma region Tasks
void New_Task_5(FarnebackParams params, bool isWrite = true, bool isFirstVideo = true, bool isView = false) {
    // Открываем видеофайл
    VideoCapture cap;
    string videoPath = isFirstVideo ?
        "C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/20201113_1516_LeftCam.mp4" :
        "C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/20191119_1241_Cam_1_03_00.mp4";

    cap.open(videoPath);
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл: " << videoPath << endl;
        return;
    }

    // Получаем параметры видео
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    cout << "Видео: " << videoPath << endl;
    cout << "Размер: " << width << "x" << height << ", FPS: " << fps << endl;
    cout << "Параметры алгоритма: " << params.nameParams << endl;

    // Создаем объекты для записи видео
    VideoWriter writer_color, writer_vector, writer_result;
    string baseName = string(isFirstVideo ? "lab5_video1_" : "lab5_video2_") + params.nameParams;

    if (isWrite) {
        // Видео с цветным представлением оптического потока
        writer_color.open(GLOBAL_RESULTS_PATH + baseName + "_color.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            computeROISize(Size(width, height)));
        if (!writer_color.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для цветного представления." << endl;
            return;
        }

        // Видео с векторным представлением оптического потока
        writer_vector.open(GLOBAL_RESULTS_PATH + baseName + "_vector.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            computeROISize(Size(width, height)));
        if (!writer_vector.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для векторного представления." << endl;
            return;
        }

        writer_result.open(GLOBAL_RESULTS_PATH + baseName + "_result.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            Size(width, height));
        if (!writer_result.isOpened()) {
            cerr << "Ошибка: не удалось создать видеофайл для векторного представления." << endl;
            return;
        }

        cout << "Созданы видеофайлы для записи." << endl;
    }

    // Создаем окно для отображения
    const string winName = "Detect_object";
    if (isView) {
        namedWindow(winName, WINDOW_NORMAL);
        resizeWindow(winName, min(1200, width), min(1000, height));
    }

    // Переменные для обработки
    Mat prevGray, currGray, frame;
    Mat flow;
    int frameNum = 0;

    // Создаем файл статистики
    ofstream statsFile(GLOBAL_RESULTS_PATH + baseName + "_stats.txt");
    if (!statsFile.is_open()) {
        cerr << "Ошибка: не удалось создать файл статистики" << endl;
        return;
    }

    // Записываем заголовок в файл статистики
    statsFile << "frame_num,processing_time_ms,min_mag,max_mag,mean_mag,dev_below,dev_above" << endl;

    // Вектор для хранения статистики по времени
    vector<long long> processingTimes;

    // Область интереса
    Rect roi;
    bool roiComputed = false;

    cout << "Начало обработки кадров..." << endl;

    // Основной цикл обработки видео
    while (true) {
        // Засекаем время начала обработки кадра
        auto startTime = high_resolution_clock::now();

        // Считываем кадр
        if (!cap.read(frame)) {
            break;
        }

        // Пропускаем первый кадр (нужен для сравнения со следующим)
        if (frameNum == 0) {
            cvtColor(frame, prevGray, COLOR_BGR2GRAY);
            frameNum++;
            continue;
        }

        // Преобразуем текущий кадр в полутоновый
        cvtColor(frame, currGray, COLOR_BGR2GRAY);

        // Вычисляем область интереса (один раз для первого кадра с оптическим потоком)
        if (!roiComputed) {
            roi = computeROI(frame);
            roiComputed = true;
            cout << "Область интереса: x=" << roi.x << ", y=" << roi.y
                << ", w=" << roi.width << ", h=" << roi.height << endl;
        }

        // Обрезаем изображения до ROI
        Mat prevGrayROI = prevGray(roi);
        Mat currGrayROI = currGray(roi);

        // Вычисляем оптический поток только в ROI по алгоритму Фарнебэка
        calcOpticalFlowFarneback(
            prevGrayROI, currGrayROI, flow,
            params.pyrScale, params.levels, params.winsize,
            params.iterations, params.polyN, params.polySigma,
            params.flags
        );

        // Вычисляем статистику по области интереса
        double minMag, maxMag, meanMag;
        int devBelow, devAbove;
        computeFlowStats(flow, minMag, maxMag, meanMag, devBelow, devAbove, 0.2);

        // Создаем визуализации ТОЛЬКО для ROI
        Mat colorFlowROI = drawOpticalFlowColor(flow);  // Цветное представление
        Mat vectorFlowROI = drawOpticalFlowVectors(flow, 20);  // Векторное представление

        // Выделяем движущиеся объекты в ROI
        Mat motionDetection = highlightMovingObjects(frame, flow, roi, minMag, maxMag, meanMag, 0.2);

        // Засекаем время окончания обработки
        auto endTime = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(endTime - startTime);
        long long processingTime = duration.count();
        processingTimes.push_back(processingTime);

        // Записываем статистику в файл
        statsFile << frameNum << "," << processingTime << ","
            << minMag << "," << maxMag << "," << meanMag << ","
            << devBelow << "," << devAbove << endl;

        // Записываем кадры в видеофайлы
        if (isWrite) {
            writer_color.write(colorFlowROI);      // Цветное представление
            writer_vector.write(vectorFlowROI);    // Векторное представление
            writer_result.write(motionDetection);
        }

        // Отображаем результат
        if (isView) {
            imshow(winName, motionDetection);
            imshow("vect", vectorFlowROI);
            imshow("color", colorFlowROI);
        }
        // Обновляем предыдущий кадр
        currGray.copyTo(prevGray);
        frameNum++;

        // Выход по нажатию ESC
        if (waitKey(1) == 27) {
            cout << "Обработка прервана пользователем." << endl;
            break;
        }

        // Выводим прогресс каждые 50 кадров
        if (frameNum % 50 == 0) {
            cout << "Обработано кадров: " << frameNum << endl;
        }
    }

    // Закрываем все ресурсы
    destroyAllWindows();
    cap.release();
    if (isWrite) {
        writer_color.release();
        writer_vector.release();
        writer_result.release();
    }
    statsFile.close();
    cout << "Обработка завершена. Обработано кадров: " << frameNum << endl;
}
#pragma endregion

int main()
{
    setlocale(LC_ALL, "Russian");

    // Определяем наборы параметров согласно заданию
    FarnebackParams paramSets;
    bool isView = false;
    bool isWrite = true;

    cout << "\n=== Набор параметров 1 ===" << endl;
    paramSets = { 0.5, 3, 15, 3, 5, 1.2, 0, "params1" };
    New_Task_5(paramSets, isWrite, true, isView);
    New_Task_5(paramSets, isWrite, false, isView);

    cout << "\n=== Набор параметров 2 ===" << endl;
    paramSets = { 0.5, 1, 15, 3, 5, 1.2, 0, "params2" };
    New_Task_5(paramSets, isWrite, true, isView);
    New_Task_5(paramSets, isWrite, false, isView);


    cout << "\n=== Набор параметров 3 ===" << endl;
    paramSets = { 0.5, 1, 15, 1, 5, 1.2, 0, "params3" };
    New_Task_5(paramSets, isWrite, true, isView);
    New_Task_5(paramSets, isWrite, false, isView);

    cout << "\n=== Все обработки завершены ===" << endl;
    return 0;
}