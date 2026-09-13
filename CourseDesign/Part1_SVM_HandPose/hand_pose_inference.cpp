#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <sys/stat.h>
#define stat _stat
#else
#include <dirent.h>
#include <sys/stat.h>
#endif
using namespace cv;
using namespace std;

// ============================================================================
// Model path config (模型的相对路径)
// ============================================================================
static string getExeDir() {
    string f = __FILE__;
    size_t pos = f.find_last_of("\\/");
    return pos == string::npos ? "." : f.substr(0, pos);
}
const string MODEL_PATH     = getExeDir() + "/hand_pose_svm_model.yml";
const string PARAMS_PATH    = getExeDir() + "/hand_pose_svm_model_params.yml";
const string LABEL_MAP[4]  = { "A", "C", "Five", "V" };

// ============================================================================
// Test data path (修改这里即可更改测试图片文件夹)
// ============================================================================
const string TEST_FOLDER    = "E:/Hand_Posture_Easy_Stu";
//
// ============================================================================
// HOG config (must match training exactly)
// ============================================================================
struct HOGConfig {
    Size winSize, blockSize, blockStride, cellSize;
    int numBins, numLevels;
    double winSigma, L2HysThresh;
    bool gammaCorrect;
    int nBins;
};
const HOGConfig HOG_CFG = {
    Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8),
    9, 1, -1.0, 0.2, false, 9
};

// ============================================================================
// Utilities
// ============================================================================
bool fileExists(const string& path) {
#ifdef _WIN32
    struct _stat st;
    return _stat(path.c_str(), &st) == 0;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0;
#endif
}

// ============================================================================
// Image preprocessing (must match training exactly)
// ============================================================================
Mat preprocessHand(const Mat& src) {
    Mat gray;
    if (src.channels() == 3) cvtColor(src, gray, COLOR_BGR2GRAY);
    else gray = src.clone();

    Mat gammaImg;
    float gamma = 0.8f;
    Mat lookupTable(1, 256, CV_8UC1);
    uchar* lut = lookupTable.ptr<uchar>();
    for (int i = 0; i < 256; i++)
        lut[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    LUT(gray, lookupTable, gammaImg);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gammaImg, gray);
    GaussianBlur(gray, gray, Size(5, 5), 1.2);

    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1, -1), 2);

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    int maxArea = 0, maxIdx = 0;
    for (int i = 1; i < numLabels; i++) {
#if CV_MAJOR_VERSION >= 4
        int area = stats.at<int>(i, CC_STAT_AREA);
#else
        int area = stats.at<int>(i, 4);
#endif
        if (area > maxArea) { maxArea = area; maxIdx = i; }
    }

    Mat cleanMask = Mat::zeros(binary.size(), CV_8UC1);
    if (maxArea > 500) cleanMask = (labels == maxIdx);

    Mat result = Mat::zeros(gray.size(), CV_8UC1);
    gray.copyTo(result, cleanMask);
    return result;
}

// ============================================================================
// Feature extraction
// ============================================================================
vector<float> extractHOG(const Mat& img) {
    Mat resized; resize(img, resized, HOG_CFG.winSize);
    if (resized.empty()) return vector<float>();
    HOGDescriptor hog(HOG_CFG.winSize, HOG_CFG.blockSize, HOG_CFG.blockStride,
                      HOG_CFG.cellSize, HOG_CFG.numBins);
    hog.winSigma = HOG_CFG.winSigma;
    hog.L2HysThreshold = HOG_CFG.L2HysThresh;
    hog.gammaCorrection = HOG_CFG.gammaCorrect;
    vector<float> desc; vector<Point> loc;
    hog.compute(resized, desc, HOG_CFG.blockStride, Size(0, 0), loc);
    return desc.empty() ? vector<float>() : desc;
}

vector<float> extractHuMoments(const Mat& img) {
    Mat binary; threshold(img, binary, 1, 255, THRESH_BINARY);
    Moments m = moments(binary, true);
    double hu[7]; HuMoments(m, hu);
    vector<float> f(7);
    for (int i = 0; i < 7; i++) {
        double val = hu[i];
        if (val == 0) val = 1e-12;
        f[i] = static_cast<float>(-copysign(1.0, val) * log10(fabs(val) + 1e-12));
    }
    return f;
}

vector<float> extractContourFeatures(const Mat& binaryImg) {
    vector<float> feat(8, 0.0f);
    vector<vector<Point>> contours;
    findContours(binaryImg.clone(), contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return feat;

    int maxIdx = 0, maxArea = 0;
    for (int i = 0; i < (int)contours.size(); i++) {
        double a = contourArea(contours[i]);
        if (a > maxArea) { maxArea = (int)a; maxIdx = i; }
    }
    const vector<Point>& c = contours[maxIdx];

    feat[0] = static_cast<float>(contourArea(c));
    feat[1] = static_cast<float>(arcLength(c, true));
    vector<Point> hull; convexHull(c, hull);
    double hullArea = contourArea(hull);
    feat[2] = static_cast<float>(hullArea);
    feat[3] = (hullArea > 1e-6f) ? static_cast<float>(maxArea / hullArea) : 0.0f;

    RotatedRect rr = minAreaRect(c);
    Size2f rrSize = rr.size;
    feat[4] = (rrSize.height > 1e-6f) ? rrSize.width / rrSize.height : 0.0f;
    feat[5] = (maxArea > 1e-6f) ? static_cast<float>((feat[1] * feat[1]) / maxArea) : 0.0f;

    Point2f center; float radius;
    minEnclosingCircle(c, center, radius);
    feat[6] = (radius > 1e-6f) ? static_cast<float>(maxArea / (CV_PI * radius * radius)) : 0.0f;

    Rect bbox = boundingRect(c);
    feat[7] = (bbox.height > 0) ? static_cast<float>(bbox.width) / static_cast<float>(bbox.height) : 0.0f;
    return feat;
}

vector<float> extractLBP(const Mat& grayImg) {
    Mat resized; resize(grayImg, resized, Size(64, 64));
    if (resized.empty()) return vector<float>();

    int rows = resized.rows, cols = resized.cols;
    Mat lbpImage(rows - 2, cols - 2, CV_8UC1);
    for (int r = 1; r < rows - 1; r++) {
        for (int c = 1; c < cols - 1; c++) {
            uchar center = resized.at<uchar>(r, c);
            uchar code = 0;
            code |= (resized.at<uchar>(r - 1, c - 1) >= center) << 7;
            code |= (resized.at<uchar>(r - 1, c    ) >= center) << 6;
            code |= (resized.at<uchar>(r - 1, c + 1) >= center) << 5;
            code |= (resized.at<uchar>(r    , c + 1) >= center) << 4;
            code |= (resized.at<uchar>(r + 1, c + 1) >= center) << 3;
            code |= (resized.at<uchar>(r + 1, c    ) >= center) << 2;
            code |= (resized.at<uchar>(r + 1, c - 1) >= center) << 1;
            code |= (resized.at<uchar>(r    , c - 1) >= center) << 0;
            lbpImage.at<uchar>(r - 1, c - 1) = code;
        }
    }

    // Rotation-invariant uniform pattern LBP (same as training)
    static int lbp2bin[256];
    static bool init = false;
    if (!init) {
        for (int v = 0; v < 256; v++) lbp2bin[v] = -1;
        for (int v = 0; v < 256; v++) {
            int bc = 0;
            for (int b = 0; b < 8; b++) {
                int curr = (v >> b) & 1;
                int prev = (v >> ((b + 7) % 8)) & 1;
                if (curr != prev) bc++;
            }
            if (bc <= 2) {
                int minRot = v;
                for (int r = 1; r < 8; r++) {
                    int rot = ((v << r) & 0xFF) | ((v >> (8 - r)));
                    if (rot < minRot) minRot = rot;
                }
                lbp2bin[v] = minRot % 58;
            } else {
                lbp2bin[v] = 58;
            }
        }
        init = true;
    }

    int nBins = 59, grid = 4;
    int cellH = lbpImage.rows / grid, cellW = lbpImage.cols / grid;
    vector<float> hist;

    for (int gx = 0; gx < grid; gx++) {
        for (int gy = 0; gy < grid; gy++) {
            Mat cell = lbpImage(Rect(gy * cellW, gx * cellH, cellW, cellH));
            vector<float> cellHist(nBins, 0.0f);
            for (int r = 0; r < cell.rows; r++) {
                for (int c = 0; c < cell.cols; c++) {
                    int binIdx = lbp2bin[cell.at<uchar>(r, c)];
                    if (binIdx >= 0 && binIdx < nBins) cellHist[binIdx]++;
                }
            }
            float sum = (float)countNonZero(cell);
            if (sum > 0) for (int b = 0; b < nBins; b++) cellHist[b] /= sum;
            hist.insert(hist.end(), cellHist.begin(), cellHist.end());
        }
    }
    return hist;
}

vector<float> extractAllFeatures(const Mat& img) {
    Mat processed = preprocessHand(img);

    auto normalizeL2 = [](vector<float>& feat) {
        float sumSq = 0;
        for (float v : feat) sumSq += v * v;
        if (sumSq > 1e-8f) {
            float norm = sqrt(sumSq);
            for (float& v : feat) v /= norm;
        }
    };

    vector<float> hog = extractHOG(processed);
    vector<float> hu = extractHuMoments(processed);
    vector<float> contour = extractContourFeatures(processed);
    vector<float> lbp = extractLBP(processed);

    normalizeL2(hog);
    normalizeL2(hu);
    normalizeL2(contour);
    normalizeL2(lbp);

    vector<float> feat;
    feat.insert(feat.end(), hog.begin(), hog.end());
    feat.insert(feat.end(), hu.begin(), hu.end());
    feat.insert(feat.end(), contour.begin(), contour.end());
    feat.insert(feat.end(), lbp.begin(), lbp.end());
    return feat;
}

// ============================================================================
// Predict single image
// ============================================================================
string predictImage(const Mat& img, Ptr<cv::ml::SVM>& svm,
                    const Mat& meanVals, const Mat& stdVals,
                    const Mat& pcaMean, const Mat& pcaVectors) {
    vector<float> feat = extractAllFeatures(img);
    Mat sample(1, (int)feat.size(), CV_32FC1);
    for (int i = 0; i < (int)feat.size(); i++)
        sample.at<float>(0, i) = feat[i];

    Mat centered = sample - pcaMean;
    Mat projected = centered * pcaVectors.t();
    Mat scaled = (projected - repeat(meanVals, 1, 1)) / repeat(stdVals, 1, 1);

    int labelIdx = (int)svm->predict(scaled);
    if (labelIdx < 0 || labelIdx >= 4) labelIdx = 0;
    return LABEL_MAP[labelIdx];
}

// Extract true label from subfolder path (parent folder name)
string extractTrueLabel(const string& filePath) {
    size_t lastSlash = filePath.find_last_of("\\/");
    if (lastSlash == string::npos) return "Unknown";
    string parent = filePath.substr(0, lastSlash);
    size_t prevSlash = parent.find_last_of("\\/");
    if (prevSlash == string::npos) return "Unknown";
    return parent.substr(prevSlash + 1);
}

// ============================================================================
// Collect all image files from a folder
// ============================================================================
// Collect all image files from a folder recursively (including subfolders)
void collectImageFilesRecursive(const string& folder, vector<string>& files) {
#ifdef _WIN32
    string searchPattern = folder + "\\*";
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &findData);
    if (hFind == INVALID_HANDLE_VALUE) return;
    do {
        string name = findData.cFileName;
        if (name == "." || name == "..") continue;
        string fullPath = folder + "\\" + name;
        struct _stat st;
        if (_stat(fullPath.c_str(), &st) == 0) {
            if (st.st_mode & _S_IFDIR) {
                collectImageFilesRecursive(fullPath, files);
            } else {
                string ext = name.size() >= 4 ? name.substr(name.size() - 4) : "";
                if (ext == ".png" || ext == ".jpg" || ext == ".PNG" || ext == ".JPG")
                    files.push_back(fullPath);
            }
        }
    } while (FindNextFileA(hFind, &findData));
    FindClose(hFind);
#else
    DIR* dir = opendir(folder.c_str());
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir))) {
        string name = entry->d_name;
        if (name == "." || name == "..") continue;
        string fullPath = folder + "/" + name;
        struct stat st;
        if (stat(fullPath.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                collectImageFilesRecursive(fullPath, files);
            } else {
                string ext = name.size() >= 4 ? name.substr(name.size() - 4) : "";
                if (ext == ".png" || ext == ".jpg" || ext == ".PNG" || ext == ".JPG")
                    files.push_back(fullPath);
            }
        }
    }
    closedir(dir);
#endif
}

vector<string> collectImageFiles(const string& folder) {
    vector<string> files;
    collectImageFilesRecursive(folder, files);
    sort(files.begin(), files.end());
    return files;
}

// ============================================================================
// Main
// ============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    cout << "================================================================================" << endl;
    cout << "          Hand Pose Recognition - Inference Script" << endl;
    cout << "          Classes: A / C / Five / V" << endl;
    cout << "================================================================================" << endl;

    // Check model files
    if (!fileExists(MODEL_PATH)) {
        cerr << "[ERROR] Model file not found: " << MODEL_PATH << endl;
        cerr << "Place hand_pose_svm_model.yml in the same directory as the executable." << endl;
        return -1;
    }
    if (!fileExists(PARAMS_PATH)) {
        cerr << "[ERROR] Params file not found: " << PARAMS_PATH << endl;
        cerr << "Place hand_pose_svm_model_params.yml in the same directory as the executable." << endl;
        return -1;
    }

    // Load SVM model
    Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(MODEL_PATH);
    cout << "[OK] SVM model loaded: " << MODEL_PATH << endl;

    // Load PCA and scaling parameters
    FileStorage fs(PARAMS_PATH, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "[ERROR] Cannot open params file: " << PARAMS_PATH << endl;
        return -1;
    }
    Mat meanVals, stdVals, pcaMean, pcaVectors;
    fs["mean"] >> meanVals;
    fs["std"] >> stdVals;
    fs["pcaMean"] >> pcaMean;
    fs["pcaVectors"] >> pcaVectors;
    fs.release();
    cout << "[OK] PCA params loaded: " << PARAMS_PATH << endl;
    cout << "     PCA dim: " << pcaVectors.rows << " x " << pcaVectors.cols << endl;

    // Get test data path (常量已在文件顶部 TEST_FOLDER 定义)
    string testFolder = TEST_FOLDER;
    if (!fileExists(testFolder)) {
        cerr << "[ERROR] Folder does not exist: " << testFolder << endl;
        return -1;
    }

    // Collect images
    vector<string> imageFiles = collectImageFiles(testFolder);
    if (imageFiles.empty()) {
        cerr << "[ERROR] No image files found in: " << testFolder << endl;
        return -1;
    }

    cout << "\nFound " << imageFiles.size() << " images in: " << testFolder << endl;
    cout << "Starting prediction..." << endl;
    cout << "================================================================================" << endl;
    cout << "                              Prediction Results" << endl;
    cout << "================================================================================" << endl;
    cout << left << setw(6) << "Index"
         << left << setw(20) << "Filename"
         << left << setw(12) << "True"
         << left << setw(12) << "Prediction"
         << left << setw(6) << "OK?" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;

    int correct = 0, total = 0;
    for (size_t i = 0; i < imageFiles.size(); i++) {
        Mat img = imread(imageFiles[i], IMREAD_COLOR);
        if (img.empty()) {
            cerr << "[WARNING] Cannot read: " << imageFiles[i] << endl;
            continue;
        }

        string result = predictImage(img, svm, meanVals, stdVals, pcaMean, pcaVectors);
        string trueLabel = extractTrueLabel(imageFiles[i]);
        bool isCorrect = (result == trueLabel);
        if (isCorrect) correct++;
        total++;

        cout << left << setw(6) << (i + 1)
             << left << setw(20) << imageFiles[i].substr(imageFiles[i].find_last_of("\\/") + 1)
             << left << setw(12) << trueLabel
             << left << setw(12) << result
             << left << setw(6) << (isCorrect ? "OK" : "FAIL") << endl;
    }

    cout << "================================================================================" << endl;
    double accuracy = (total > 0) ? (100.0 * correct / total) : 0.0;
    cout << fixed << setprecision(2);
    cout << "Accuracy: " << correct << "/" << total << " = " << accuracy << "%" << endl;
    cout << "================================================================================" << endl;

    return 0;
}
