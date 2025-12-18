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

string GLOBAL_RESULTS_PATH = "C:/Users/ilyah/Desktop/MGY/IS_RTK/result_6/";

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

Mat improvedAutoCanny(const Mat& gray, double sigma = 0.33) {
    // 1. Добавляем размытие для уменьшения шума
    Mat blurred;
    GaussianBlur(gray, blurred, Size(3, 3), 0);

    // 2. Вычисляем медиану интенсивности с учетом только градиентов
    Mat grad_x, grad_y;
    Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
    Sobel(blurred, grad_y, CV_32F, 0, 1, 3);

    Mat magnitude;
    magnitude = abs(grad_x) + abs(grad_y);

    // 3. Нормализуем и вычисляем медиану
    magnitude.convertTo(magnitude, CV_8U);
    double v = computeMedian(magnitude);

    // 4. Вычисляем пороги
    int lower = static_cast<int>(max(0.0, (1.0 - sigma) * v));
    int upper = static_cast<int>(min(255.0, (1.0 + sigma) * v));

    // 5. Увеличиваем верхний порог для лучшей связности
    upper = min(255, upper * 2);

    Mat edges;
    Canny(blurred, edges, lower, upper);

    return edges;
}

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

// Функция для сравнения контуров по близости в последовательных кадрах
double contourDistance(const vector<Point>& contour1, const vector<Point>& contour2) {
    Moments m1 = moments(contour1);
    Moments m2 = moments(contour2);

    Point2f center1(m1.m10 / m1.m00, m1.m01 / m1.m00);
    Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);

    double area1 = contourArea(contour1);
    double area2 = contourArea(contour2);

    // Комбинированное расстояние: учитываем и расстояние между центрами, и разницу площадей
    double dist = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    double areaDiff = abs(area1 - area2) / max(area1, area2);

    return dist + areaDiff * 100; // Весовой коэффициент для разницы площадей
}
#pragma endregion

#pragma region Tasks
void New_Task_6(bool isWrite = true, bool isView = false, bool isCanny = true) {
    // Открываем видеофайл
    VideoCapture cap;
    string videoPath = "C:/Users/ilyah/Desktop/MGY/IS_RTK/Video/Moving object.mp4";
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

    // Создаем объекты для записи видео
    VideoWriter writer_result;
    string baseName = "lab6" + string(isCanny ? "_canny" : "_threshold");

    if (isWrite) {
        writer_result.open(GLOBAL_RESULTS_PATH + baseName + "_result.avi",
            VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps,
            Size(width, height));
        if (!writer_result.isOpened()) {
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
    Mat frame, gray, edges;
    int frameNum = 0;

    // Параметры для фильтрации объектов
    double minArea = 1000.0;
    double length_px = 180.0;
    double max_length_px = 250.0;
    double trackingThreshold = 150.0; // Максимальное расстояние для отслеживания между кадрами
    int maxTrackHistory = 70; // Максимальная длина истории трека

    vector<Point> lastTrackedContour;
    vector<Point2f> trackHistory; // История позиций центра для отрисовки трека
    bool isTracking = false;
 

    // Создаем файл статистики
    ofstream statsFile(GLOBAL_RESULTS_PATH + baseName + "_stats.txt");
    if (!statsFile.is_open()) {
        cerr << "Ошибка: не удалось создать файл статистики" << endl;
        return;
    }

    // Записываем заголовок в файл статистики
    statsFile << "Frame ProcessingTime(ms) NumObjects CenterX CenterY BBoxWidth BBoxHeight Tracked" << endl;

    cout << "Начало обработки кадров..." << endl;

    // Основной цикл обработки видео
    while (true) {
        // Засекаем время начала обработки кадра
        auto startTime = high_resolution_clock::now();

        // Считываем кадр
        if (!cap.read(frame)) {
            break;
        }

        Mat resultFrame = frame.clone();

        // Преобразуем в оттенки серого
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        //GaussianBlur(gray, gray, Size(7, 7), 0, 0);

        // Применяем детектор Кэнни
        if (isCanny) {
            edges = adaptiveCanny(gray, 0.33);
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
            morphologyEx(edges, edges, MORPH_CLOSE, kernel);
        }
        else threshold(gray, edges, 100, 255, THRESH_BINARY_INV);

        if (isView) {
            imshow(winName2, edges);
        }        
        // Находим контуры
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Фильтруем контуры по площади
        vector<vector<Point>> filteredContours;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area >= minArea) {
                filteredContours.push_back(contour);
            }
        }

        int numObjects = filteredContours.size();

        // Переменные для текущего предпочтительного объекта
        int preferredIndex = -1;
        double minAreaDiff = numeric_limits<double>::max();
        Point2f preferredCenter(0, 0);
        Rect preferredBbox(0, 0, 0, 0);
        bool trackedThisFrame = false;

        // Если есть объект, который мы отслеживали в прошлом кадре
        if (isTracking && !lastTrackedContour.empty() && !filteredContours.empty()) {
            // Ищем ближайший контур к предыдущему
            double minDistance = numeric_limits<double>::max();
            int closestIndex = -1;

            for (int i = 0; i < filteredContours.size(); i++) {
                double dist = contourDistance(lastTrackedContour, filteredContours[i]);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestIndex = i;
                }

                preferredBbox = boundingRect(filteredContours[i]);
                if ((max(abs(preferredBbox.width - length_px), abs(preferredBbox.height - length_px)) < minAreaDiff) && (max_length_px > max(preferredBbox.height, preferredBbox.width))) {
                    minAreaDiff = max(abs(preferredBbox.width - length_px), abs(preferredBbox.height - length_px));
                    preferredIndex = i;
                }
            }
            if (closestIndex != preferredIndex) {
                // Потеряли объект
                cout << "Объект пропал из виду" << endl;
                isTracking = false;
                trackedThisFrame = false;
            }

            // Если нашли достаточно близкий контур
            if (closestIndex != -1 && minDistance < trackingThreshold) {
                preferredIndex = closestIndex;
                trackedThisFrame = true;

                // Обновляем последний отслеживаемый контур
                lastTrackedContour = filteredContours[preferredIndex];

                // Вычисляем центр и рамку
                Moments m = moments(lastTrackedContour);
                if (m.m00 != 0) {
                    preferredCenter.x = m.m10 / m.m00;
                    preferredCenter.y = m.m01 / m.m00;
                }
                preferredBbox = boundingRect(lastTrackedContour);
                // Добавляем точку в историю трека
                trackHistory.push_back(preferredCenter);

                // Ограничиваем длину истории трека
                if (trackHistory.size() > maxTrackHistory) {
                    trackHistory.erase(trackHistory.begin());
                }
            }
            else {
                // Потеряли объект
                cout << "Объект пропал из виду" << endl;
                isTracking = false;
                trackedThisFrame = false;
            }
        }

        // Если не отследили объект, ищем новый по площади
        if (!trackedThisFrame) {
            // Выбираем предпочтительный объект (ближайший к ожидаемой площади)
            for (int i = 0; i < filteredContours.size(); i++) {
                preferredBbox = boundingRect(filteredContours[i]);
                if ((max(abs(preferredBbox.width - length_px), abs(preferredBbox.height - length_px)) < minAreaDiff) && (max_length_px > max(preferredBbox.height, preferredBbox.width))) {
                    minAreaDiff = max(abs(preferredBbox.width - length_px), abs(preferredBbox.height - length_px));
                    preferredIndex = i;
                }
            }

            if (preferredIndex != -1) {
                // Нашли новый объект для отслеживания
                lastTrackedContour = filteredContours[preferredIndex];
                isTracking = true;

                // Вычисляем центр и рамку
                Moments m = moments(lastTrackedContour);
                if (m.m00 != 0) {
                    preferredCenter.x = m.m10 / m.m00;
                    preferredCenter.y = m.m01 / m.m00;
                }
                preferredBbox = boundingRect(lastTrackedContour);

                // Начинаем новую историю трека
                trackHistory.clear();
                trackHistory.push_back(preferredCenter);

                trackedThisFrame = false; // Это новый объект, не отслеженный с прошлого кадра
            }
            else {
                // Не нашли подходящих объектов
                isTracking = false;
            }
        }

        // Отрисовываем все контуры (бирюзовым, толщина 1)
        for (int i = 0; i < filteredContours.size(); i++) {
            drawContours(resultFrame, filteredContours, i, Scalar(0, 0, 255), 2);
        }

        // Отрисовываем предпочтительный объект
        if (preferredIndex != -1) {
            Scalar boxColor;
            if (trackedThisFrame) {
                boxColor = Scalar(0, 255, 0); // Зеленый - успешно отслежен
            }
            else {
                boxColor = Scalar(0, 0, 255); // Красный - новый или потерянный
            }

            // Отрисовываем рамку
            rectangle(resultFrame, preferredBbox, boxColor, 3);

            // Отрисовываем центр
            circle(resultFrame, preferredCenter, 5, Scalar(255, 0, 0), -1);

            // Отрисовываем трек (желтой линией)
            if (trackHistory.size() > 1) {
                for (size_t i = 1; i < trackHistory.size(); i++) {
                    line(resultFrame, trackHistory[i - 1], trackHistory[i], Scalar(0, 255, 255), 2);
                }
            }
        }

        // Записываем в файл статистики
        auto endTime = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(endTime - startTime);
        if (preferredCenter.x == 0 && preferredCenter.y == 0) {
            statsFile << frameNum << " "
                << duration.count() << " "
                << numObjects << " "
                << 0 << " " << 0 << " "
                << 0 << " " << 0 << " "
                << (trackedThisFrame ? 1 : 0) << endl;
        }
        else {
            statsFile << frameNum << " "
                << duration.count() << " "
                << numObjects << " "
                << preferredCenter.x << " " << preferredCenter.y << " "
                << preferredBbox.width << " " << preferredBbox.height << " "
                << (trackedThisFrame ? 1 : 0) << endl;
        }
        // Записываем кадры в видеофайлы
        if (isWrite) {
            writer_result.write(resultFrame);
        }

        // Отображаем результат
        if (isView) {
            imshow(winName, resultFrame);
        }

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
        writer_result.release();
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
    bool isCanny = true;     // Режим детекции с помощью Canny/Threshold

    New_Task_6(isWrite, isView, isCanny);

    isCanny = false;
    New_Task_6(isWrite, isView, isCanny);
    return 0;
}