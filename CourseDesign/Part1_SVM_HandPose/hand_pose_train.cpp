//// ============================================================================
//// 机器视觉课程设计 — 手势识别
//// 识别 A / C / Five / V 四种手势
//// HOG + LBP + Hu矩 + 轮廓特征融合 + SVM分类
//// 训练集和测试集完全分离，自动选择最佳模型
//// ============================================================================
//
//#include <opencv2/opencv.hpp>
//#include <opencv2/ml.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <string>
//#include <sstream>
//#include <cmath>
//#include <ctime>
//#include <cstdlib>
//#include <iomanip>
//#ifdef _WIN32
//#include <windows.h>
//#include <io.h>
//#include <sys/stat.h>
//#define stat _stat
//#else
//#include <dirent.h>
//#include <sys/stat.h>
//#endif
//using namespace cv;
//using namespace std;
//
//// ============================================================================
//// 全局配置
//// ============================================================================
//const string TRAIN_ROOT = R"(E:\XieBro\大学规划\课堂课件\大二下\机器视觉\实验\课程设计\Hand_Posture_Easy_Stu\Hand_Posture_Easy_Stu)";
//const string TEST_ROOT  = R"(E:\XieBro\大学规划\课堂课件\大二下\机器视觉\实验\课程设计\Hand_Posture_Easy_Stu\Hand_Posture_Easy_Stu)";
//const string MODEL_PATH   = "hand_pose_svm_model.yml";
//const string LABEL_MAP[4] = { "A", "C", "Five", "V" };
//const string CLASS_FOLDERS[4] = { "A", "C", "Five", "V" };
//
//const int TRAIN_PER_CLASS = 45;   // 每类训练样本数
//const int TEST_PER_CLASS  = 5;    // 每类测试样本数
//const int K_FOLDS = 5;
//
//// ============================================================================
//// 数据增强配置
//// ============================================================================
//struct AugmentConfig {
//    bool enable;        // 是否启用数据增强
//    int brightnessN;    // 亮度变化次数
//    int deltaRange;     // 亮度变化幅度 [-deltaRange, +deltaRange]
//    bool flip;          // 水平翻转
//    vector<double> rotAngles;    // 旋转角度列表
//    vector<double> scales;       // 缩放比例列表
//    bool addNoise;      // 高斯噪声
//    int noiseStd;       // 噪声标准差
//};
//
//const AugmentConfig AUGMENT_CFG = {
//    true,               // enable
//    2,                  // brightnessN: 每张生成2种亮度变化
//    30,                 // deltaRange: ±30
//    true,               // flip: 水平翻转
//    {-12.0, -6.0, 6.0, 12.0},  // rotAngles: -12°/-6°/+6°/+12°
//    {0.9, 0.95, 1.05, 1.1},     // scales: 0.9x~1.1x
//    true,               // noise
//    15                  // noiseStd
//};
//
//// ============================================================================
//// HOG参数配置（可调节优化）
//// ============================================================================
//struct HOGConfig {
//    Size winSize;       // 窗口大小
//    Size blockSize;     // block大小
//    Size blockStride;   // block步长
//    Size cellSize;      // cell大小
//    int numBins;        // 方向 bins 数
//    int numLevels;      // 图像金字塔层数
//    double winSigma;    // 高斯窗口sigma（-1表示不用高斯）
//    double L2HysThresh; // L2-Hys归一化阈值
//    bool gammaCorrect;  // gamma预处理
//    int nBins;          // 直方图bins（同numBins）
//};
//
//const HOGConfig HOG_CFG = {
//    Size(128, 128),   // winSize
//    Size(16, 16),     // blockSize
//    Size(8, 8),       // blockStride
//    Size(8, 8),       // cellSize
//    9,                // numBins
//    1,                // numLevels
//    -1.0,             // winSigma（使用默认值）
//    0.2,              // L2HysThresh
//    false,            // gammaCorrect
//    9                 // nBins（同numBins）
//};
//
//// ============================================================================
//// PCA/模型配置
//// ============================================================================
//const int PCA_TARGET_DIM = 120;  // PCA目标维度（50→120）
//
////============================================================================
////功能开关
////============================================================================
//// #define SHOW_PREVIEW     // 启用则显示图像预览（仅测试模式）
//// #define VERBOSE_FEATURES // 启用则输出每张图的特征维度
//
//// ============================================================================
//// 工具函数
//// ============================================================================
//bool fileExists(const string& path) {
//#ifdef _WIN32
//    struct _stat st;
//    return _stat(path.c_str(), &st) == 0;
//#else
//    struct stat st;
//    return stat(path.c_str(), &st) == 0;
//#endif
//}
//
//// ============================================================================
//// 图像预处理
//// ============================================================================
//Mat preprocessHand(const Mat& src) {
//    Mat gray;
//    if (src.channels() == 3) {
//        cvtColor(src, gray, COLOR_BGR2GRAY);
//    } else {
//        gray = src.clone();
//    }
//
//    Mat gammaImg;
//    float gamma = 0.8f;
//    Mat lookupTable(1, 256, CV_8UC1);
//    uchar* lut = lookupTable.ptr<uchar>();
//    for (int i = 0; i < 256; i++) {
//        lut[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
//    }
//    LUT(gray, lookupTable, gammaImg);
//
//    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
//    clahe->apply(gammaImg, gray);
//    GaussianBlur(gray, gray, Size(5, 5), 1.2);
//
//    Mat binary;
//    threshold(gray, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
//
//    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
//    morphologyEx(binary, binary, MORPH_CLOSE, kernel, Point(-1, -1), 2);
//
//    Mat labels, stats, centroids;
//    int numLabels = connectedComponentsWithStats(binary, labels, stats, centroids, 8);
//    int maxArea = 0, maxIdx = 0;
//    for (int i = 1; i < numLabels; i++) {
//#if CV_MAJOR_VERSION >= 4
//        int area = stats.at<int>(i, CC_STAT_AREA);
//#else
//        int area = stats.at<int>(i, 4);
//#endif
//        if (area > maxArea) { maxArea = area; maxIdx = i; }
//    }
//
//    Mat cleanMask = Mat::zeros(binary.size(), CV_8UC1);
//    if (maxArea > 500) cleanMask = (labels == maxIdx);
//
//    Mat result = Mat::zeros(gray.size(), CV_8UC1);
//    gray.copyTo(result, cleanMask);
//    return result;
//}
//
//// ============================================================================
//// 数据增强：基于配置生成多种变换版本
//// ============================================================================
//vector<Mat> augmentImage(const Mat& src, const AugmentConfig& cfg) {
//    vector<Mat> variants;
//    if (src.empty()) return variants;
//
//    Mat graySrc = src;
//    if (src.channels() == 3) {
//        Mat tmp;
//        cvtColor(src, tmp, COLOR_BGR2GRAY);
//        graySrc = tmp;
//    }
//
//    RNG rng(getTickCount());
//
//    // 亮度变化
//    for (int v = 0; v < cfg.brightnessN; v++) {
//        int delta = rng.uniform(-cfg.deltaRange, cfg.deltaRange + 1);
//        Mat bright;
//        graySrc.convertTo(bright, -1, 1.0, delta);
//        variants.push_back(bright);
//    }
//
//    // 水平翻转
//    if (cfg.flip) {
//        Mat flipped;
//        flip(graySrc, flipped, 1);
//        variants.push_back(flipped);
//    }
//
//    // 旋转
//    Point2f center(graySrc.cols / 2.0f, graySrc.rows / 2.0f);
//    for (double angle : cfg.rotAngles) {
//        Mat rot = getRotationMatrix2D(center, angle, 1.0);
//        Mat rotated;
//        warpAffine(graySrc, rotated, rot, graySrc.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
//        variants.push_back(rotated);
//    }
//
//    // 缩放
//    for (double scale : cfg.scales) {
//        Mat scaled;
//        resize(graySrc, scaled, Size(), scale, scale, INTER_LINEAR);
//        int roiX = max(0, (graySrc.cols - scaled.cols) / 2);
//        int roiY = max(0, (graySrc.rows - scaled.rows) / 2);
//        int roiW = min(scaled.cols, graySrc.cols);
//        int roiH = min(scaled.rows, graySrc.rows);
//        if (roiW > 0 && roiH > 0) {
//            Mat padded = Mat::zeros(graySrc.size(), graySrc.type());
//            Rect roi(roiX, roiY, roiW, roiH);
//            scaled(Rect(0, 0, roiW, roiH)).copyTo(padded(roi));
//            variants.push_back(padded);
//        }
//    }
//
//    // 高斯噪声
//    if (cfg.addNoise) {
//        Mat noise = graySrc.clone();
//        rng.fill(noise, RNG::NORMAL, Scalar(0), Scalar(cfg.noiseStd));
//        Mat noisy;
//        add(graySrc, noise, noisy);
//        variants.push_back(noisy);
//    }
//
//    return variants;
//}
//
//// ============================================================================
//// 特征提取
//// ============================================================================
//vector<float> extractHOG(const Mat& img, const HOGConfig& cfg) {
//    Mat resized;
//    resize(img, resized, cfg.winSize);
//    if (resized.empty()) return vector<float>();
//
//    HOGDescriptor hog(
//        cfg.winSize,     // 检测窗口大小
//        cfg.blockSize,   // block大小
//        cfg.blockStride, // block步长
//        cfg.cellSize,    // cell大小
//        cfg.numBins      // 方向bins
//    );
//    hog.winSigma = cfg.winSigma;
//    hog.L2HysThreshold = cfg.L2HysThresh;
//    hog.gammaCorrection = cfg.gammaCorrect;
//
//    vector<float> descriptors;
//    vector<Point> locations;
//    hog.compute(resized, descriptors, cfg.blockStride, Size(0, 0), locations);
//    return descriptors.empty() ? vector<float>() : descriptors;
//}
//
//// 重载：使用默认HOG配置
//vector<float> extractHOG(const Mat& img) {
//    return extractHOG(img, HOG_CFG);
//}
//
//vector<float> extractHuMoments(const Mat& img) {
//    Mat binary;
//    threshold(img, binary, 1, 255, THRESH_BINARY);
//    Moments m = moments(binary, true);
//    double hu[7];
//    HuMoments(m, hu);
//    vector<float> features(7);
//    for (int i = 0; i < 7; i++) {
//        double val = hu[i];
//        if (val == 0) val = 1e-12;
//        features[i] = static_cast<float>(-copysign(1.0, val) * log10(fabs(val) + 1e-12));
//    }
//    return features;
//}
//
//vector<float> extractContourFeatures(const Mat& binaryImg) {
//    vector<float> feat(8, 0.0f);
//    vector<vector<Point>> contours;
//    findContours(binaryImg.clone(), contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//    if (contours.empty()) return feat;
//
//    int maxIdx = 0, maxArea = 0;
//    for (int i = 0; i < (int)contours.size(); i++) {
//        double a = contourArea(contours[i]);
//        if (a > maxArea) { maxArea = (int)a; maxIdx = i; }
//    }
//    const vector<Point>& c = contours[maxIdx];
//
//    feat[0] = static_cast<float>(contourArea(c));
//    feat[1] = static_cast<float>(arcLength(c, true));
//    vector<Point> hull;
//    convexHull(c, hull);
//    double hullArea = contourArea(hull);
//    feat[2] = static_cast<float>(hullArea);
//    feat[3] = (hullArea > 1e-6f) ? static_cast<float>(maxArea / hullArea) : 0.0f;
//
//    RotatedRect rr = minAreaRect(c);
//    Size2f rrSize = rr.size;
//    feat[4] = (rrSize.height > 1e-6f) ? rrSize.width / rrSize.height : 0.0f;
//    feat[5] = (maxArea > 1e-6f) ? static_cast<float>((feat[1] * feat[1]) / maxArea) : 0.0f;
//
//    Point2f center;
//    float radius;
//    minEnclosingCircle(c, center, radius);
//    feat[6] = (radius > 1e-6f) ? static_cast<float>(maxArea / (CV_PI * radius * radius)) : 0.0f;
//
//    Rect bbox = boundingRect(c);
//    feat[7] = (bbox.height > 0) ? static_cast<float>(bbox.width) / static_cast<float>(bbox.height) : 0.0f;
//    return feat;
//}
//
//vector<float> extractLBP(const Mat& grayImg) {
//    Mat resized;
//    resize(grayImg, resized, Size(64, 64));
//    if (resized.empty()) return vector<float>();
//
//    int rows = resized.rows, cols = resized.cols;
//    Mat lbpImage(rows - 2, cols - 2, CV_8UC1);
//    for (int r = 1; r < rows - 1; r++) {
//        for (int c = 1; c < cols - 1; c++) {
//            uchar center = resized.at<uchar>(r, c);
//            uchar code = 0;
//            code |= (resized.at<uchar>(r - 1, c - 1) >= center) << 7;
//            code |= (resized.at<uchar>(r - 1, c    ) >= center) << 6;
//            code |= (resized.at<uchar>(r - 1, c + 1) >= center) << 5;
//            code |= (resized.at<uchar>(r    , c + 1) >= center) << 4;
//            code |= (resized.at<uchar>(r + 1, c + 1) >= center) << 3;
//            code |= (resized.at<uchar>(r + 1, c    ) >= center) << 2;
//            code |= (resized.at<uchar>(r + 1, c - 1) >= center) << 1;
//            code |= (resized.at<uchar>(r    , c - 1) >= center) << 0;
//            lbpImage.at<uchar>(r - 1, c - 1) = code;
//        }
//    }
//
//    static int lbp2bin[256];
//    static bool init = false;
//    if (!init) {
//        for (int v = 0; v < 256; v++) lbp2bin[v] = -1;
//        for (int v = 0; v < 256; v++) {
//            int bc = 0;
//            for (int b = 0; b < 8; b++) {
//                int curr = (v >> b) & 1;
//                int prev = (v >> ((b + 7) % 8)) & 1;
//                if (curr != prev) bc++;
//            }
//            if (bc <= 2) {
//                int minRot = v;
//                for (int r = 1; r < 8; r++) {
//                    int rot = ((v << r) & 0xFF) | ((v >> (8 - r)));
//                    if (rot < minRot) minRot = rot;
//                }
//                lbp2bin[v] = minRot % 58;
//            } else {
//                lbp2bin[v] = 58;
//            }
//        }
//        init = true;
//    }
//
//    int nBins = 59, grid = 4;
//    int cellH = lbpImage.rows / grid, cellW = lbpImage.cols / grid;
//    vector<float> hist;
//
//    for (int gx = 0; gx < grid; gx++) {
//        for (int gy = 0; gy < grid; gy++) {
//            Mat cell = lbpImage(Rect(gy * cellW, gx * cellH, cellW, cellH));
//            vector<float> cellHist(nBins, 0.0f);
//            for (int r = 0; r < cell.rows; r++) {
//                for (int c = 0; c < cell.cols; c++) {
//                    int binIdx = lbp2bin[cell.at<uchar>(r, c)];
//                    if (binIdx >= 0 && binIdx < nBins) cellHist[binIdx]++;
//                }
//            }
//            float sum = (float)countNonZero(cell);
//            if (sum > 0) for (int b = 0; b < nBins; b++) cellHist[b] /= sum;
//            hist.insert(hist.end(), cellHist.begin(), cellHist.end());
//        }
//    }
//    return hist;
//}
//
//// ============================================================================
//// 特征提取 - 改进版：使用归一化的特征 + 可配置HOG
//// ============================================================================
//vector<float> extractNormalizedFeatures(const Mat& src) {
//    static int logFlag = 0;
//    Mat preprocessed = preprocessHand(src);
//    if (preprocessed.empty()) {
//        cerr << "[WARNING] preprocessHand returned empty image" << endl;
//        return vector<float>();
//    }
//
//    Mat binary;
//    threshold(preprocessed, binary, 1, 255, THRESH_BINARY);
//
//    // 提取各特征
//    vector<float> hogFeat = extractHOG(preprocessed);
//    vector<float> huFeat = extractHuMoments(preprocessed);
//    vector<float> cntFeat = extractContourFeatures(binary);
//    vector<float> lbpFeat = extractLBP(preprocessed);
//
//    if (logFlag++ == 0 && !hogFeat.empty()) {
//        cout << "    [HOG配置] win=(" << HOG_CFG.winSize.width << "x" << HOG_CFG.winSize.height << ") "
//             << "block=(" << HOG_CFG.blockSize.width << "x" << HOG_CFG.blockSize.height << ") "
//             << "cell=(" << HOG_CFG.cellSize.width << "x" << HOG_CFG.cellSize.height << ") "
//             << "stride=(" << HOG_CFG.blockStride.width << "x" << HOG_CFG.blockStride.height << ") "
//             << "bins=" << HOG_CFG.numBins << " | HOG维度=" << hogFeat.size() << endl;
//    }
//
//    // 确保所有特征非空
//    if (hogFeat.empty()) hogFeat = vector<float>(8100, 0.0f);
//    if (huFeat.empty()) huFeat = vector<float>(7, 0.0f);
//    if (cntFeat.empty()) cntFeat = vector<float>(8, 0.0f);
//    if (lbpFeat.empty()) lbpFeat = vector<float>(944, 0.0f);
//
//    // L2归一化每种特征
//    auto normalizeL2 = [](vector<float>& feat) {
//        float sumSq = 0;
//        for (float v : feat) sumSq += v * v;
//        if (sumSq > 1e-8f) {
//            float norm = sqrt(sumSq);
//            for (float& v : feat) v /= norm;
//        }
//    };
//
//    normalizeL2(hogFeat);
//    normalizeL2(huFeat);
//    normalizeL2(cntFeat);
//    normalizeL2(lbpFeat);
//
//    // 合并特征
//    vector<float> all;
//    all.reserve(hogFeat.size() + huFeat.size() + cntFeat.size() + lbpFeat.size());
//    all.insert(all.end(), hogFeat.begin(), hogFeat.end());
//    all.insert(all.end(), huFeat.begin(), huFeat.end());
//    all.insert(all.end(), cntFeat.begin(), cntFeat.end());
//    all.insert(all.end(), lbpFeat.begin(), lbpFeat.end());
//
//    return all;
//}
//
//// 兼容旧函数名
//vector<float> extractAllFeatures(const Mat& src) {
//    return extractNormalizedFeatures(src);
//}
//
//// ============================================================================
//// 数据集加载（训练集和测试集完全分离）
//// ============================================================================
//
//// 收集某类文件夹下所有图片路径（排序后返回）
//vector<string> collectImagePaths(const string& root, int classIdx) {
//    vector<string> pngFiles;
//#ifdef _WIN32
//    string folder = root + "\\" + CLASS_FOLDERS[classIdx];
//    string searchPattern = folder + "\\*";
//    WIN32_FIND_DATAA findData;
//    HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &findData);
//    if (hFind == INVALID_HANDLE_VALUE || !hFind) return pngFiles;
//    do {
//        string name = findData.cFileName;
//        if (name.size() >= 4) {
//            string ext = name.substr(name.size() - 4);
//            if (ext == ".png" || ext == ".jpg" || ext == ".PNG" || ext == ".JPG")
//                pngFiles.push_back(folder + "\\" + name);
//        }
//    } while (FindNextFileA(hFind, &findData));
//    FindClose(hFind);
//#else
//    string folder = root + "/" + CLASS_FOLDERS[classIdx];
//    DIR* hFind = opendir(folder.c_str());
//    struct dirent* entry = NULL;
//    if (!hFind) return pngFiles;
//    while ((entry = readdir(hFind)) != NULL) {
//        string name = entry->d_name;
//        if (name.size() >= 4) {
//            string ext = name.substr(name.size() - 4);
//            if (ext == ".png" || ext == ".jpg" || ext == ".PNG" || ext == ".JPG")
//                pngFiles.push_back(folder + "/" + name);
//        }
//    }
//    closedir(hFind);
//#endif
//    sort(pngFiles.begin(), pngFiles.end());
//    return pngFiles;
//}
//
//// 分层随机划分：每类随机选 TEST_PER_CLASS 张作为测试集，其余 TRAIN_PER_CLASS 张作为训练集
//// 所有随机操作使用全局 srand(42) 保证可复现
//struct StratifiedSplit {
//    vector<vector<int>> trainIndices;  // trainIndices[classIdx] = 该类训练集在 pngFiles 中的索引
//    vector<vector<int>> testIndices;   // testIndices[classIdx]  = 该类测试集在 pngFiles 中的索引
//};
//
//StratifiedSplit createStratifiedSplit(const string& root) {
//    StratifiedSplit split;
//    split.trainIndices.resize(4);
//    split.testIndices.resize(4);
//
//    cout << "\n--- 分层随机划分 (srand=42) ---" << endl;
//    for (int classIdx = 0; classIdx < 4; classIdx++) {
//        vector<string> pngFiles = collectImagePaths(root, classIdx);
//        int total = (int)pngFiles.size();
//
//        vector<int> allIndices(total);
//        for (int i = 0; i < total; i++) allIndices[i] = i;
//
//        // Fisher-Yates 洗牌（用 rand() 因为 srand(42) 已在 main 中设置）
//        for (int i = total - 1; i > 0; i--) {
//            int j = rand() % (i + 1);
//            swap(allIndices[i], allIndices[j]);
//        }
//
//        // 前 TEST_PER_CLASS 放入测试集，后面的放入训练集
//        for (int i = 0; i < TEST_PER_CLASS && i < total; i++)
//            split.testIndices[classIdx].push_back(allIndices[i]);
//        for (int i = TEST_PER_CLASS; i < total; i++)
//            split.trainIndices[classIdx].push_back(allIndices[i]);
//
//        cout << "  [" << LABEL_MAP[classIdx] << "] 共" << total << "张 | "
//             << "测试集索引: ";
//        for (size_t t = 0; t < split.testIndices[classIdx].size(); t++)
//            cout << split.testIndices[classIdx][t] << (t < split.testIndices[classIdx].size() - 1 ? "," : "");
//        cout << " | 训练集: " << (int)split.trainIndices[classIdx].size() << "张" << endl;
//    }
//    return split;
//}
//
//// 加载训练集：接收预划分好的训练索引，对每张原图做数据增强
//void loadTrainDataset(vector<Mat>& trainImages, vector<int>& trainLabels,
//                      const string& root, const AugmentConfig& augCfg,
//                      const StratifiedSplit& split) {
//    // 计算增强倍数（通过试运行获取）
//    int augMultiplier = 0;
//    if (augCfg.enable) {
//        Mat dummyImg(100, 100, CV_8UC3, Scalar(128, 128, 128));
//        vector<Mat> dummyVariants = augmentImage(dummyImg, augCfg);
//        augMultiplier = (int)dummyVariants.size();
//    }
//
//    int totalOrig = 0;
//    for (int classIdx = 0; classIdx < 4; classIdx++) {
//        vector<string> pngFiles = collectImagePaths(root, classIdx);
//        const vector<int>& indices = split.trainIndices[classIdx];
//
//        int origLoaded = 0;
//        for (int idx : indices) {
//            if (idx < 0 || idx >= (int)pngFiles.size()) continue;
//            Mat img = imread(pngFiles[idx], IMREAD_COLOR);
//            if (!img.empty()) {
//                trainImages.push_back(img);
//                trainLabels.push_back(classIdx);
//                origLoaded++;
//
//                if (augCfg.enable) {
//                    vector<Mat> variants = augmentImage(img, augCfg);
//                    for (const Mat& v : variants) {
//                        trainImages.push_back(v);
//                        trainLabels.push_back(classIdx);
//                    }
//                }
//            }
//        }
//        totalOrig += origLoaded;
//        int totalPerClass = origLoaded * (augCfg.enable ? (1 + augMultiplier) : 1);
//        cout << "  [" << LABEL_MAP[classIdx] << "] "
//             << origLoaded << " 原图";
//        if (augCfg.enable)
//            cout << " + " << (origLoaded * augMultiplier) << " 增强 = " << totalPerClass << " 张";
//        cout << endl;
//    }
//    int totalPerImg = augCfg.enable ? (1 + augMultiplier) : 1;
//    cout << "  [总计] " << totalOrig << " 原图/类 x " << totalPerImg << " = "
//         << (totalOrig * totalPerImg / 4) * 4 << " 总训练样本" << endl;
//}
//
//// 加载测试集：接收预划分好的测试索引，只加载原图（无增强）
//void loadTestDataset(vector<Mat>& testImages, vector<int>& testLabels,
//                     const string& root, const StratifiedSplit& split) {
//    for (int classIdx = 0; classIdx < 4; classIdx++) {
//        vector<string> pngFiles = collectImagePaths(root, classIdx);
//        const vector<int>& indices = split.testIndices[classIdx];
//
//        int loaded = 0;
//        for (int idx : indices) {
//            if (idx < 0 || idx >= (int)pngFiles.size()) continue;
//            Mat img = imread(pngFiles[idx], IMREAD_COLOR);
//            if (!img.empty()) {
//                testImages.push_back(img);
//                testLabels.push_back(classIdx);
//                loaded++;
//            }
//        }
//        cout << "  [" << LABEL_MAP[classIdx] << "] 加载 " << loaded << " 张测试图像" << endl;
//    }
//}
//
//// ============================================================================
//// 特征矩阵构建
//// ============================================================================
//Mat featuresToMat(const vector<Mat>& images) {
//    if (images.empty()) {
//        cerr << "[ERROR] featuresToMat: images vector is empty" << endl;
//        return Mat();
//    }
//
//    cout << "提取第一张图像特征..." << endl;
//    vector<float> firstFeat = extractAllFeatures(images[0]);
//    int dim = (int)firstFeat.size();
//    if (dim == 0) {
//        cerr << "[ERROR] featuresToMat: first image feature dimension is 0" << endl;
//        return Mat();
//    }
//    cout << "特征维度: " << dim << endl;
//
//    Mat data((int)images.size(), dim, CV_32FC1, Scalar::all(0));
//    for (int j = 0; j < dim; j++) data.at<float>(0, j) = firstFeat[j];
//
//    for (size_t i = 1; i < images.size(); i++) {
//        vector<float> feat = extractAllFeatures(images[i]);
//        if ((int)feat.size() != dim) {
//            cerr << "[WARNING] Image " << i << " feature dim mismatch (got " << feat.size()
//                 << ", expected " << dim << "), filling with zeros" << endl;
//            feat.resize(dim, 0.0f);
//        }
//        for (int j = 0; j < dim; j++) data.at<float>((int)i, j) = feat[j];
//    }
//    return data;
//}
//
//// ============================================================================
//// 数据标准化 + PCA
//// ============================================================================
//struct PreparedData {
//    Mat scaledData, meanVals, stdVals, pcaData, pcaMean, pcaVectors;
//    PCA pca;
//    int actualDim;
//};
//
//PreparedData prepareData(const Mat& rawData) {
//    PreparedData pd;
//
//    // 错误检查
//    if (rawData.empty()) {
//        cerr << "[ERROR] prepareData: rawData is empty!" << endl;
//        return pd;
//    }
//    if (rawData.rows < 10) {
//        cerr << "[ERROR] prepareData: too few samples (" << rawData.rows << ")" << endl;
//        return pd;
//    }
//
//    cout << "原始数据: " << rawData.rows << " samples x " << rawData.cols << " dims" << endl;
//
//    // Step 1: 先PCA降维（处理高维小样本问题）
//    Mat dataF;
//    rawData.convertTo(dataF, CV_32FC1);
//
//    // PCA目标维度：从50提升至120，保留更多判别信息
//    int maxPcaDim = min(rawData.rows - 1, PCA_TARGET_DIM);
//    cout << "PCA目标维度: " << maxPcaDim << " (配置值=" << PCA_TARGET_DIM << ")" << endl;
//
//    // PCA保留99%能量或达到目标维度
//    pd.pca = PCA(dataF, Mat(), PCA::DATA_AS_ROW, 0.99);
//    Mat pcaProjected = pd.pca.project(dataF);
//
//    int actualPcaDim = min(pcaProjected.cols, maxPcaDim);
//    if (actualPcaDim > 0 && actualPcaDim < pcaProjected.cols) {
//        pcaProjected = pcaProjected(Rect(0, 0, actualPcaDim, pcaProjected.rows));
//    }
//    cout << "PCA后维度: " << pcaProjected.cols << endl;
//
//    // 保存PCA参数
//    pd.pcaMean = pd.pca.mean.reshape(1, 1);
//    pd.pcaVectors = pd.pca.eigenvectors.rowRange(0, actualPcaDim);
//
//    // Step 2: 对PCA降维后的数据进行标准化
//    reduce(pcaProjected, pd.meanVals, 0, REDUCE_AVG);
//    Mat sqDiffs = pcaProjected - repeat(pd.meanVals, pcaProjected.rows, 1);
//    multiply(sqDiffs, sqDiffs, sqDiffs);
//    reduce(sqDiffs, pd.stdVals, 0, REDUCE_AVG);
//    sqrt(pd.stdVals, pd.stdVals);
//    for (int i = 0; i < pd.stdVals.cols; i++)
//        if (pd.stdVals.at<float>(0, i) < 1e-8f) pd.stdVals.at<float>(0, i) = 1.0f;
//
//    pd.pcaData = (pcaProjected - repeat(pd.meanVals, pcaProjected.rows, 1))
//                 / repeat(pd.stdVals, pcaProjected.rows, 1);
//
//    pd.actualDim = pd.pcaData.cols;
//    pd.scaledData = pd.pcaData.clone();
//
//    cout << "最终维度: " << pd.actualDim << endl;
//
//    return pd;
//}
//
//// ============================================================================
//// K折交叉验证划分（分层抽样）
//// ============================================================================
//vector<vector<int>> createKFoldIndices(const Mat& labelsMat, int k, int numClasses) {
//    int n = labelsMat.rows;
//    vector<vector<int>> class_indices(numClasses);
//    for (int i = 0; i < n; i++) {
//        int label = labelsMat.at<int>(i);
//        if (label >= 0 && label < numClasses) class_indices[label].push_back(i);
//    }
//
//    for (int c = 0; c < numClasses; c++) {
//        vector<int>& vec = class_indices[c];
//        for (int i = (int)vec.size() - 1; i > 0; i--) {
//            int j = rand() % (i + 1);
//            swap(vec[i], vec[j]);
//        }
//    }
//
//    vector<vector<int>> fold_indices(k);
//    for (int f = 0; f < k; f++) {
//        for (int c = 0; c < numClasses; c++) {
//            const vector<int>& vec = class_indices[c];
//            int perFold = (int)vec.size() / k;
//            int start = f * perFold;
//            int end = (f == k - 1) ? (int)vec.size() : (f + 1) * perFold;
//            for (int i = start; i < end; i++) fold_indices[f].push_back(vec[i]);
//        }
//    }
//    return fold_indices;
//}
//
//// ============================================================================
//// SVM Kernel配置
//// ============================================================================
//struct KernelConfig {
//    int kernelType;
//    string name;
//    double gamma;
//    double degree;
//    double coef0;
//};
//
//vector<KernelConfig> getKernelConfigs() {
//    vector<KernelConfig> configs;
//
//    // LINEAR kernel - 细粒度C搜索
//    configs.push_back({cv::ml::SVM::LINEAR, "LINEAR", 0, 0, 0});
//
//    // RBF kernel - 关键kernel，细粒度搜索
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=0.001)", 0.001, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=0.01)", 0.01, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=0.05)", 0.05, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=0.1)", 0.1, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=0.5)", 0.5, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=1)", 1.0, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=5)", 5.0, 0, 0});
//    configs.push_back({cv::ml::SVM::RBF, "RBF(gamma=auto)", -1, 0, 0});  // -1表示auto
//
//    // POLY kernel
//    configs.push_back({cv::ml::SVM::POLY, "POLY(deg=2)", 1.0, 2.0, 0.0});
//    configs.push_back({cv::ml::SVM::POLY, "POLY(deg=3)", 1.0, 3.0, 0.0});
//    configs.push_back({cv::ml::SVM::POLY, "POLY(deg=4)", 1.0, 4.0, 0.0});
//
//    // SIGMOID kernel
//    configs.push_back({cv::ml::SVM::SIGMOID, "SIGMOID", 0.01, 0, 0});
//    configs.push_back({cv::ml::SVM::SIGMOID, "SIGMOID", 0.1, 0, 0});
//    configs.push_back({cv::ml::SVM::SIGMOID, "SIGMOID", 1.0, 0, 0});
//
//    return configs;
//}
//
//// ============================================================================
//// K折交叉验证
//// ============================================================================
//struct CVResult {
//    int kernelType;
//    string kernelName;
//    double C;
//    double avgAccuracy;
//    double stdAccuracy;
//    vector<double> foldAccuracies;
//    double trainingTime;
//    int numSupportVectors;
//};
//
//CVResult crossValidateWithKernel(const Mat& pcaData, const Mat& labelsMat,
//                                  int kernelType, const string& kernelName,
//                                  double C, double gamma, double degree, double coef0,
//                                  int k, int numClasses) {
//    CVResult result;
//    result.kernelType = kernelType;
//    result.kernelName = kernelName;
//    result.C = C;
//    result.foldAccuracies.clear();
//    result.avgAccuracy = 0.0;
//    result.stdAccuracy = 0.0;
//    result.trainingTime = 0.0;
//    result.numSupportVectors = 0;
//
//    // 错误检查
//    if (pcaData.empty() || labelsMat.empty()) {
//        cerr << "[ERROR] crossValidate: empty data!" << endl;
//        return result;
//    }
//
//    vector<vector<int>> fold_indices = createKFoldIndices(labelsMat, k, numClasses);
//    double totalCorrect = 0;
//    int n = pcaData.rows;
//
//    clock_t totalStart = clock();
//
//    for (int f = 0; f < k; f++) {
//        vector<int> valIdx = fold_indices[f];
//        vector<int> trainIdx;
//        for (int ff = 0; ff < k; ff++) {
//            if (ff != f) for (int id : fold_indices[ff]) trainIdx.push_back(id);
//        }
//
//        // 检查是否有足够的数据
//        if (trainIdx.size() < 2 || valIdx.size() < 1) {
//            cerr << "[ERROR] crossValidate: not enough samples for fold " << f << endl;
//            continue;
//        }
//
//        Mat trainData((int)trainIdx.size(), pcaData.cols, CV_32FC1);
//        Mat valData((int)valIdx.size(), pcaData.cols, CV_32FC1);
//        Mat trainLabels((int)trainIdx.size(), 1, CV_32SC1);
//        Mat valLabels((int)valIdx.size(), 1, CV_32SC1);
//
//        for (size_t i = 0; i < trainIdx.size(); i++) {
//            pcaData.row(trainIdx[i]).copyTo(trainData.row((int)i));
//            trainLabels.at<int>((int)i) = labelsMat.at<int>(trainIdx[i]);
//        }
//        for (size_t i = 0; i < valIdx.size(); i++) {
//            pcaData.row(valIdx[i]).copyTo(valData.row((int)i));
//            valLabels.at<int>((int)i) = labelsMat.at<int>(valIdx[i]);
//        }
//
//        Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
//        svm->setType(cv::ml::SVM::C_SVC);
//        svm->setKernel(kernelType);
//        svm->setC(C);
//        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-6));
//
//        if (kernelType == cv::ml::SVM::POLY) {
//            svm->setDegree(degree);
//            svm->setGamma(gamma);
//            svm->setCoef0(coef0);
//        } else if (kernelType == cv::ml::SVM::RBF) {
//            svm->setGamma(gamma == -1 ? 1.0 / pcaData.cols : gamma);
//        } else if (kernelType == cv::ml::SVM::SIGMOID) {
//            svm->setGamma(gamma);
//            svm->setCoef0(coef0);
//        }
//
//        clock_t t0 = clock();
//        try {
//            svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);
//        } catch (const cv::Exception& e) {
//            cerr << "[ERROR] SVM train failed: " << e.what() << endl;
//            continue;
//        }
//        clock_t t1 = clock();
//        result.trainingTime += (double)(t1 - t0) / CLOCKS_PER_SEC;
//
//        Mat sv = svm->getSupportVectors();
//        result.numSupportVectors += sv.rows;
//
//        int correct = 0;
//        for (int i = 0; i < valData.rows; i++) {
//            int pred = (int)svm->predict(valData.row(i));
//            if (pred == valLabels.at<int>(i)) correct++;
//        }
//
//        double foldAcc = (double)correct / valData.rows * 100.0;
//        result.foldAccuracies.push_back(foldAcc);
//        totalCorrect += correct;
//    }
//
//    if (result.foldAccuracies.empty()) {
//        cerr << "[ERROR] crossValidate: all folds failed!" << endl;
//        return result;
//    }
//
//    result.avgAccuracy = totalCorrect / n * 100.0;
//    result.trainingTime /= result.foldAccuracies.size();
//    result.numSupportVectors /= result.foldAccuracies.size();
//
//    double sumSq = 0;
//    for (double acc : result.foldAccuracies)
//        sumSq += (acc - result.avgAccuracy) * (acc - result.avgAccuracy);
//    result.stdAccuracy = sqrt(sumSq / result.foldAccuracies.size());
//
//    return result;
//}
//
//// ============================================================================
//// 分类器结果
//// ============================================================================
//struct ClassifierResult {
//    string name;
//    int kernelType;
//    double C;
//    double gamma;
//    double degree;
//    double cvAcc;
//    double stdAcc;
//    double trainTime;
//    int numSV;
//    vector<double> foldAccs;
//};
//
//struct BestModelInfo {
//    string name;
//    int kernelType;
//    double C;
//    double gamma;
//    double degree;
//    double cvAcc;
//    double stdAcc;
//};
//
//// ============================================================================
//// 对比所有SVM配置
//// ============================================================================
//vector<ClassifierResult> compareAllClassifiers(const Mat& pcaData, const Mat& labelsMat) {
//    cout << "\n================================================================================" << endl;
//    cout << "               " << K_FOLDS << "折交叉验证 - SVM多Kernel对比" << endl;
//    cout << "================================================================================" << endl;
//    cout << "样本数: " << pcaData.rows << " | PCA维度: " << pcaData.cols << " | 折数: " << K_FOLDS << endl << endl;
//
//    vector<ClassifierResult> allResults;
//    vector<KernelConfig> kernels = getKernelConfigs();
//    // C参数细化搜索：分三个区间细粒度搜索
//    vector<double> C_values = {
//        0.001, 0.005,   // 细粒度搜索：小值区间
//        0.01, 0.05, 0.1, 0.5,  // 中值区间
//        1.0, 5.0, 10.0,  // 常用区间
//        25.0, 50.0,  // 大值细化
//        100.0, 250.0, 500.0  // 大值区间
//    };
//
//    clock_t startAll = clock();
//    int totalConfigs = (int)(kernels.size() * C_values.size());
//    int currentConfig = 0;
//
//    cout << "开始对比 " << totalConfigs << " 种配置..." << endl << endl;
//
//    for (const auto& kernel : kernels) {
//        for (double C : C_values) {
//            currentConfig++;
//            cout << "[" << currentConfig << "/" << totalConfigs << "] "
//                 << kernel.name << " | C=" << C << " ... " << flush;
//
//            CVResult cvResult = crossValidateWithKernel(
//                pcaData, labelsMat, kernel.kernelType, kernel.name,
//                C, kernel.gamma, kernel.degree, kernel.coef0, K_FOLDS, 4);
//
//            ClassifierResult r;
//            r.name = kernel.name + "-C" + to_string(C);
//            r.kernelType = kernel.kernelType;
//            r.C = C;
//            r.gamma = kernel.gamma;
//            r.degree = kernel.degree;
//            r.cvAcc = cvResult.avgAccuracy;
//            r.stdAcc = cvResult.stdAccuracy;
//            r.trainTime = cvResult.trainingTime;
//            r.numSV = cvResult.numSupportVectors;
//            r.foldAccs = cvResult.foldAccuracies;
//            allResults.push_back(r);
//
//            cout << fixed << setprecision(2) << cvResult.avgAccuracy << "% (std="
//                 << cvResult.stdAccuracy << "%)" << endl;
//        }
//    }
//
//    sort(allResults.begin(), allResults.end(),
//         [](const ClassifierResult& a, const ClassifierResult& b) { return a.cvAcc > b.cvAcc; });
//
//    cout << "\n================================================================================" << endl;
//    cout << "                          交叉验证结果排名" << endl;
//    cout << "================================================================================" << endl;
//    cout << left << setw(4) << "排名" << setw(5) << " "
//         << setw(25) << "配置" << setw(12) << "平均准确率" << setw(12) << "标准差"
//         << setw(10) << "训练时间" << setw(8) << "支持向量" << endl;
//    cout << "--------------------------------------------------------------------------------" << endl;
//
//    for (size_t i = 0; i < allResults.size(); i++) {
//        cout << right << setw(3) << (i + 1) << ". "
//             << left << setw(25) << allResults[i].name
//             << fixed << setprecision(2)
//             << setw(10) << " " << setw(10) << allResults[i].cvAcc << "%"
//             << setw(12) << allResults[i].stdAcc << "%"
//             << setw(8) << allResults[i].trainTime << "s"
//             << setw(8) << allResults[i].numSV << endl;
//    }
//
//    cout << "\n================================================================================" << endl;
//    cout << "                    最优模型 [" << allResults[0].name << "] 各折详情" << endl;
//    cout << "================================================================================" << endl;
//    cout << "各折准确率: ";
//    for (size_t f = 0; f < allResults[0].foldAccs.size(); f++) {
//        cout << fixed << setprecision(2) << allResults[0].foldAccs[f] << "%";
//        if (f < allResults[0].foldAccs.size() - 1) cout << " / ";
//    }
//    cout << endl;
//    cout << "平均准确率: " << fixed << setprecision(2) << allResults[0].cvAcc << "%" << endl;
//    cout << "标准差: " << fixed << setprecision(2) << allResults[0].stdAcc << "%" << endl;
//
//    cout << "\n>>> 最优分类器: [" << allResults[0].name << "] | "
//         << K_FOLDS << "折CV准确率: " << allResults[0].cvAcc << "%" << endl;
//    cout << "总用时: " << (double)(clock() - startAll) / CLOCKS_PER_SEC << "s" << endl;
//
//    return allResults;
//}
//
//// ============================================================================
//// 训练最优模型
//// ============================================================================
//BestModelInfo trainBestClassifier(const Mat& pcaData, const Mat& labelsMat,
//                                  const ClassifierResult& best,
//                                  const Mat& meanVals, const Mat& stdVals,
//                                  const Mat& pcaMean, const Mat& pcaVectors, int actualDim) {
//    cout << "\n================================================================================" << endl;
//    cout << "                 最终模型训练: " << best.name << endl;
//    cout << "================================================================================" << endl;
//
//    Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
//    svm->setType(cv::ml::SVM::C_SVC);
//    svm->setKernel(best.kernelType);
//    svm->setC(best.C);
//    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-6));
//
//    if (best.kernelType == cv::ml::SVM::POLY) {
//        svm->setDegree(best.degree);
//        svm->setGamma(best.gamma);
//    } else if (best.kernelType == cv::ml::SVM::RBF) {
//        svm->setGamma(best.gamma == -1 ? 1.0 / pcaData.cols : best.gamma);
//    } else if (best.kernelType == cv::ml::SVM::SIGMOID) {
//        svm->setGamma(best.gamma);
//        svm->setCoef0(0.0);
//    }
//
//    cout << "Kernel类型: " << best.kernelType << endl;
//    cout << "正则化参数 C: " << best.C << endl;
//
//    clock_t t0 = clock();
//    svm->train(pcaData, cv::ml::ROW_SAMPLE, labelsMat);
//    clock_t t1 = clock();
//    cout << "训练用时: " << (double)(t1 - t0) / CLOCKS_PER_SEC << "s" << endl;
//
//    svm->save(MODEL_PATH);
//    cout << "SVM模型已保存: " << MODEL_PATH << endl;
//
//    Ptr<cv::ml::SVM> loaded = cv::ml::SVM::load(MODEL_PATH);
//    int trainCorrect = 0;
//    for (int i = 0; i < pcaData.rows; i++) {
//        if ((int)loaded->predict(pcaData.row(i)) == labelsMat.at<int>(i)) trainCorrect++;
//    }
//    double trainAcc = (double)trainCorrect / pcaData.rows * 100.0;
//    cout << "[验证] 训练集准确率: " << trainAcc << "% (" << trainCorrect << "/" << pcaData.rows << ")" << endl;
//
//    Mat meanF, stdF, pcaVecF, pcaMeanF;
//    meanVals.convertTo(meanF, CV_32FC1);
//    stdVals.convertTo(stdF, CV_32FC1);
//    pcaVectors.convertTo(pcaVecF, CV_32FC1);
//    pcaMean.convertTo(pcaMeanF, CV_32FC1);
//
//    string pcaPath = MODEL_PATH.substr(0, MODEL_PATH.rfind('.')) + "_params.yml";
//    FileStorage fs(pcaPath, FileStorage::WRITE);
//    fs << "model_type" << best.name;
//    fs << "kernel_type" << best.kernelType;
//    fs << "svm_C" << best.C;
//    fs << "gamma" << best.gamma;
//    fs << "degree" << best.degree;
//    fs << "cv_accuracy" << best.cvAcc;
//    fs << "cv_std" << best.stdAcc;
//    fs << "mean" << meanF;
//    fs << "std" << stdF;
//    fs << "pcaMean" << pcaMeanF;
//    fs << "pcaVectors" << pcaVecF;
//    fs << "actualDim" << actualDim;
//    fs.release();
//    cout << "参数已保存: " << pcaPath << endl;
//
//    BestModelInfo info;
//    info.name = best.name;
//    info.kernelType = best.kernelType;
//    info.C = best.C;
//    info.gamma = best.gamma;
//    info.degree = best.degree;
//    return info;
//}
//
//// ============================================================================
//// 训练入口
//// ============================================================================
//BestModelInfo runTraining(Mat trainData, const vector<int>& trainLabels) {
//    cout << "\n================================================================================" << endl;
//    cout << "                          特征提取中..." << endl;
//    cout << "================================================================================" << endl;
//
//    Mat labelsMat((int)trainLabels.size(), 1, CV_32SC1);
//    for (size_t i = 0; i < trainLabels.size(); i++)
//        labelsMat.at<int>((int)i) = trainLabels[i];
//
//    cout << "原始特征维度: " << trainData.cols << endl;
//
//    PreparedData pd = prepareData(trainData);
//    cout << "PCA降维后维度: " << pd.actualDim << " (上限" << PCA_TARGET_DIM << ")" << endl;
//    cout << "训练矩阵: " << pd.pcaData.size() << endl;
//
//    vector<ClassifierResult> allResults = compareAllClassifiers(pd.pcaData, labelsMat);
//
//    BestModelInfo info = trainBestClassifier(
//        pd.pcaData, labelsMat, allResults[0],
//        pd.meanVals, pd.stdVals, pd.pca.mean,
//        pd.pca.eigenvectors.rowRange(0, pd.actualDim), pd.actualDim);
//
//    info.cvAcc = allResults[0].cvAcc;
//    info.stdAcc = allResults[0].stdAcc;
//    return info;
//}
//
//// ============================================================================
//// 单张图像预测
//// ============================================================================
//string predictImage(const string& modelPath, const Mat& img) {
//    vector<float> feat = extractAllFeatures(img);
//    Mat sample(1, (int)feat.size(), CV_32FC1);
//    for (int i = 0; i < (int)feat.size(); i++) sample.at<float>(0, i) = feat[i];
//
//    string pcaPath = modelPath.substr(0, modelPath.rfind('.')) + "_params.yml";
//    FileStorage fs(pcaPath, FileStorage::READ);
//    string modelType;
//    Mat meanVals, stdVals, pcaMean, pcaVectors;
//    fs["model_type"] >> modelType;
//    fs["mean"] >> meanVals;
//    fs["std"] >> stdVals;
//    fs["pcaMean"] >> pcaMean;
//    fs["pcaVectors"] >> pcaVectors;
//    fs.release();
//
//    // Step 1: PCA投影
//    Mat centered = sample - pcaMean;
//    Mat projected = centered * pcaVectors.t();
//
//    // Step 2: 标准化
//    Mat scaled = (projected - repeat(meanVals, 1, 1)) / repeat(stdVals, 1, 1);
//
//    Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
//    int labelIdx = (int)svm->predict(scaled);
//
//    if (labelIdx < 0 || labelIdx >= 4) labelIdx = 0;
//    return LABEL_MAP[labelIdx];
//}
//
//// ============================================================================
//// 训练模式
//// ============================================================================
//void runTrainMode() {
//    cout << "================================================================================" << endl;
//    cout << "                              训练模式" << endl;
//    cout << "================================================================================" << endl;
//    cout << "训练集: " << TRAIN_ROOT << endl;
//    cout << "测试集: " << TEST_ROOT << endl;
//    cout << "每类: " << TRAIN_PER_CLASS << " 训练 / " << TEST_PER_CLASS << " 测试" << endl;
//    cout << "交叉验证: " << K_FOLDS << " 折" << endl;
//
//    vector<Mat> trainImages, testImages;
//    vector<int> trainLabels, testLabels;
//
//    // 先做分层随机划分（所有随机操作的根基，确保可复现）
//    StratifiedSplit split = createStratifiedSplit(TRAIN_ROOT);
//
//    cout << "\n--- 加载训练集 (增强 " << (AUGMENT_CFG.enable ? "启用" : "禁用") << ") ---" << endl;
//    loadTrainDataset(trainImages, trainLabels, TRAIN_ROOT, AUGMENT_CFG, split);
//
//    cout << "\n--- 加载测试集 ---" << endl;
//    loadTestDataset(testImages, testLabels, TEST_ROOT, split);
//
//    cout << "\n加载完成: 训练 " << trainImages.size()
//         << " 张, 测试 " << testImages.size() << " 张" << endl;
//
//    if (trainImages.empty()) {
//        cerr << "[ERROR] No training images loaded! Check TRAIN_ROOT path." << endl;
//        return;
//    }
//    if (testImages.empty()) {
//        cerr << "[ERROR] No test images loaded! Check TEST_ROOT path." << endl;
//        return;
//    }
//
//    cout << "\n================================================================================" << endl;
//    cout << "                          特征提取中..." << endl;
//    cout << "================================================================================" << endl;
//    Mat trainData = featuresToMat(trainImages);
//    Mat testData = featuresToMat(testImages);
//
//    if (trainData.empty() || testData.empty()) {
//        cerr << "[ERROR] Feature extraction failed!" << endl;
//        return;
//    }
//
//    cout << "训练特征: " << trainData.cols << " dims x " << trainData.rows << " samples" << endl;
//    cout << "测试特征: " << testData.cols << " dims x " << testData.rows << " samples" << endl;
//
//    // 训练：用K折CV选择最佳模型
//    BestModelInfo info = runTraining(trainData, trainLabels);
//
//    // 用最佳模型在测试集上评估（批量，不弹窗）
//    cout << "\n================================================================================" << endl;
//    cout << "                          测试集评估" << endl;
//    cout << "================================================================================" << endl;
//    cout << "模型: " << info.name << endl;
//    cout << "CV准确率: " << fixed << setprecision(2) << info.cvAcc
//         << "% (std=" << info.stdAcc << "%)" << endl;
//
//    // 提取测试集特征并用保存的模型评估
//    Mat labelsMat(testLabels, true);
//    labelsMat = labelsMat.reshape(1, (int)testLabels.size());
//    labelsMat.convertTo(labelsMat, CV_32SC1);
//
//    // 加载保存的模型和参数
//    string pcaPath = MODEL_PATH.substr(0, MODEL_PATH.rfind('.')) + "_params.yml";
//    FileStorage fs(pcaPath, FileStorage::READ);
//    if (!fs.isOpened()) {
//        cerr << "[ERROR] Cannot open model file: " << pcaPath << endl;
//        return;
//    }
//
//    Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(MODEL_PATH);
//    string modelType;
//    Mat meanVals, stdVals, pcaMean, pcaVectors;
//    fs["model_type"] >> modelType;
//    fs["mean"] >> meanVals;
//    fs["std"] >> stdVals;
//    fs["pcaMean"] >> pcaMean;
//    fs["pcaVectors"] >> pcaVectors;
//    fs.release();
//
//    // 对测试集逐个预测（必须用与训练相同的PCA流程处理）
//    int confusion[4][4] = {0};
//    int correct = 0;
//    for (int i = 0; i < testData.rows; i++) {
//        Mat rawSample = testData.row(i);
//        Mat rawSampleF;
//        rawSample.convertTo(rawSampleF, CV_32FC1);
//
//        // Step 1: 用保存的PCA参数投影
//        Mat centered = rawSampleF - pcaMean;
//        Mat projected = centered * pcaVectors.t();
//
//        // Step 2: 用保存的标准化参数缩放
//        Mat scaled = (projected - repeat(meanVals, 1, 1)) / repeat(stdVals, 1, 1);
//
//        int predicted = (int)svm->predict(scaled);
//        int actual = testLabels[i];
//        if (predicted < 0 || predicted >= 4) predicted = 0;
//        if (actual < 0 || actual >= 4) actual = 0;
//        confusion[actual][predicted]++;
//        if (predicted == actual) correct++;
//    }
//
//    double testAcc = (double)correct / testData.rows * 100.0;
//    cout << "测试集准确率: " << fixed << setprecision(2) << testAcc
//         << "% (" << correct << "/" << testData.rows << ")" << endl;
//
//    cout << "\n混淆矩阵:" << endl;
//    cout << setw(12) << "预测→" << setw(10) << "A" << setw(10) << "C"
//         << setw(10) << "Five" << setw(10) << "V" << endl;
//    for (int i = 0; i < 4; i++) {
//        cout << setw(12) << LABEL_MAP[i];
//        for (int j = 0; j < 4; j++) cout << setw(10) << confusion[i][j];
//        cout << endl;
//    }
//
//    cout << "\n各类别召回率:" << endl;
//    for (int i = 0; i < 4; i++) {
//        int sum = 0;
//        for (int j = 0; j < 4; j++) sum += confusion[i][j];
//        double recall = (sum > 0) ? (double)confusion[i][i] / sum * 100.0 : 0.0;
//        cout << "  " << LABEL_MAP[i] << ": " << fixed << setprecision(2) << recall << "%" << endl;
//    }
//
//    cout << "\n================================================================================" << endl;
//    cout << "                              完成" << endl;
//    cout << "================================================================================" << endl;
//}
//
//// ============================================================================
//// 测试模式
//// ============================================================================
//void runTestMode() {
//    string pcaPath = MODEL_PATH.substr(0, MODEL_PATH.rfind('.')) + "_params.yml";
//    if (!fileExists(MODEL_PATH) || !fileExists(pcaPath)) {
//        cerr << "[ERROR] Model file not found. Please train first." << endl;
//        return;
//    }
//
//    cout << "================================================================================" << endl;
//    cout << "                              测试模式" << endl;
//    cout << "================================================================================" << endl;
//
//    FileStorage fs(pcaPath, FileStorage::READ);
//    string modelType;
//    double cvAcc, cvStd;
//    Mat meanVals, stdVals, pcaMean, pcaVectors;
//    fs["model_type"] >> modelType;
//    fs["cv_accuracy"] >> cvAcc;
//    fs["cv_std"] >> cvStd;
//    fs["mean"] >> meanVals;
//    fs["std"] >> stdVals;
//    fs["pcaMean"] >> pcaMean;
//    fs["pcaVectors"] >> pcaVectors;
//    fs.release();
//
//    Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(MODEL_PATH);
//
//    cout << "模型: " << modelType << endl;
//    cout << "CV准确率: " << fixed << setprecision(2) << cvAcc << "% (std=" << cvStd << "%)" << endl;
//
//    cout << "\n--- 加载测试集 ---" << endl;
//    vector<Mat> testImages;
//    vector<int> testLabels;
//    StratifiedSplit split = createStratifiedSplit(TEST_ROOT);
//    loadTestDataset(testImages, testLabels, TEST_ROOT, split);
//    cout << "共加载 " << testImages.size() << " 张测试图像" << endl;
//
//    if (testImages.empty()) return;
//
//    Mat testData = featuresToMat(testImages);
//    if (testData.empty()) return;
//
//    int confusion[4][4] = {0};
//    int correct = 0;
//    for (int i = 0; i < testData.rows; i++) {
//        Mat rawSample = testData.row(i);
//        Mat rawSampleF;
//        rawSample.convertTo(rawSampleF, CV_32FC1);
//
//        Mat centered = rawSampleF - pcaMean;
//        Mat projected = centered * pcaVectors.t();
//        Mat scaled = (projected - repeat(meanVals, 1, 1)) / repeat(stdVals, 1, 1);
//
//        int predicted = (int)svm->predict(scaled);
//        int actual = testLabels[i];
//        if (predicted < 0 || predicted >= 4) predicted = 0;
//        if (actual < 0 || actual >= 4) actual = 0;
//        confusion[actual][predicted]++;
//        if (predicted == actual) correct++;
//
//#ifdef SHOW_PREVIEW
//        Mat disp;
//        resize(testImages[i], disp, Size(400, 400));
//        putText(disp, LABEL_MAP[predicted], Point(20, 50),
//                FONT_HERSHEY_SIMPLEX, 1.5,
//                (predicted == actual) ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);
//        putText(disp, "Actual: " + string(LABEL_MAP[actual]), Point(20, 90),
//                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 1);
//        imshow("Result", disp);
//        int k = waitKey(0);
//        if (k == 27) break;
//#endif
//    }
//
//    double testAcc = (double)correct / testData.rows * 100.0;
//    cout << "\n测试集准确率: " << fixed << setprecision(2) << testAcc
//         << "% (" << correct << "/" << testData.rows << ")" << endl;
//
//    cout << "\n混淆矩阵:" << endl;
//    cout << setw(12) << "预测→" << setw(10) << "A" << setw(10) << "C"
//         << setw(10) << "Five" << setw(10) << "V" << endl;
//    for (int i = 0; i < 4; i++) {
//        cout << setw(12) << LABEL_MAP[i];
//        for (int j = 0; j < 4; j++) cout << setw(10) << confusion[i][j];
//        cout << endl;
//    }
//
//    cout << "\n各类别召回率:" << endl;
//    for (int i = 0; i < 4; i++) {
//        int sum = 0;
//        for (int j = 0; j < 4; j++) sum += confusion[i][j];
//        double recall = (sum > 0) ? (double)confusion[i][i] / sum * 100.0 : 0.0;
//        cout << "  " << LABEL_MAP[i] << ": " << fixed << setprecision(2) << recall << "%" << endl;
//    }
//
//    cout << "\n================================================================================" << endl;
//    cout << "                              完成" << endl;
//    cout << "================================================================================" << endl;
//}
//
//// ============================================================================
//// 主函数
//// ============================================================================
//int main() {
//    srand(42);
//    cout << "================================================================================" << endl;
//    cout << "          手势识别 - 机器视觉课程设计" << endl;
//    cout << "          类别: A / C / Five / V" << endl;
//    cout << "          方法: HOG + LBP + Hu矩 + 轮廓特征融合 + SVM" << endl;
//    cout << "          验证: " << K_FOLDS << "折交叉验证选最优模型" << endl;
//    cout << "================================================================================" << endl;
//
//    runTrainMode();
//
//    return 0;
//}
