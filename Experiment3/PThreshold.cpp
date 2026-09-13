#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

/*
 * ================================================================
 * p率阈值化（P-Percent Thresholding）是什么？
 * ================================================================
 * 首先理解什么是"阈值化"（Thresholding）。
 *
 * 阈值化 = 把灰度图像变成只有黑白两种颜色的二值图像。
 * 设置一个"阈值 T"：
 *   - 灰度值 >= T 的像素 → 变成白色（255）
 *   - 灰度值 <  T 的像素 → 变成黑色（0）
 * 这是图像分割中最简单、最直接的方法。
 *
 * 举例：
 *   一张手写数字图像，数字是深色的，背景是浅色的。
 *   如果阈值 T = 128，那么：
 *     灰度值 0~127 的像素（深色的数字部分）→ 黑色（0）
 *     灰度值 128~255 的像素（浅色的背景）→ 白色（255）
 *   结果：数字变成了黑色，背景变成了白色，分离成功！
 *
 * 问题是：阈值 T 怎么选？
 *   - 手动选 T = 128？太武断了，不一定适合所有图像。
 *   - 全局固定阈值？对所有图像都用同一个 T？也不行。
 *   - 所以有了"自适应阈值"方法，根据图像内容自动决定 T。
 *
 * p率阈值化（P-Percent Thresholding）就是一种自适应阈值方法：
 *   找到阈值 T，使得图像中"比 T 暗（或比 T 亮）的像素数量"占总像素的 p%。
 *   比如 p = 50%，T 就是直方图的"中位数"灰度级。
 *
 * 为什么叫"p率"？
 *   p 是一个百分比值（0.0 ~ 1.0）。
 *   p=0.5 表示 50%，即把图像分成"最亮的 50% 像素"和"其余 50% 像素"。
 *
 * 与 OTSU 方法的区别：
 *   OTSU（大津法）找的是让"前景和背景的类间方差最大化"的阈值，
 *   p率阈值化找的是"累计像素数刚好达到 p% 时"的灰度级。
 *   p率方法更直观，更容易理解和控制。
 */

/*
 * myThresholdP: 实现 p 率阈值化分割
 *
 * 思路：
 *   1. 统计图像的灰度直方图（hist[k] = 灰度值 k 的像素个数）
 *   2. 计算累计直方图（CDF）
 *   3. 找到灰度级 T，使得 CDF[T] >= p * totalPixels
 *      即：所有比 T 暗的像素数量达到了总像素的 p%
 *   4. 对图像做二值化：灰度值 >= T 的像素 → 255（白），否则 → 0（黑）
 *      注意：这里的逻辑是把"较亮"的像素作为"前景"（白色）。
 *
 * 参数: img - 输入灰度图
 * 返回: 二值化后的图像（只有 0 和 255 两个灰度值）
 */
Mat myThresholdP(Mat img)
{
    Mat ThresholdPImg;  // 存储二值化结果

    /*
     * p 率参数：可调整
     * p = 0.5 表示：让图像中较亮的那 50% 像素变成白色
     * p = 0.3 表示：让图像中较亮的那 30% 像素变成白色
     * p = 0.7 表示：让图像中较亮的那 70% 像素变成白色
     *
     * 举例：一幅大部分区域偏暗的图像（如夜景）
     *   p=0.5 可能让大部分像素变成白色（只有最暗的50%是黑色）
     *   p=0.3 则会让更多区域变成黑色（只有最亮的30%是白色）
     */
    double p = 0.5; // p 率，可根据需要修改（如 0.3、0.7 等）

    int height = img.rows;      // 图像高度
    int width = img.cols;       // 图像宽度
    int totalPixels = height * width;  // 总像素数量

    /*
     * 第一步：统计灰度直方图（Histogram）
     *
     * hist 是一个长度为 256 的数组，初始全为 0。
     * 然后遍历图像的每个像素，把该像素的灰度值"投到"对应的桶里。
     *
     * 举例：
     *   如果某个像素灰度值为 120，执行 hist[120]++ 后，
     *   hist[120] 就从原来的值增加 1。
     *   遍历完整个图像后：
     *     hist[0]   = 灰度值为 0 的像素总数
     *     hist[1]   = 灰度值为 1 的像素总数
     *     ...
     *     hist[255] = 灰度值为 255 的像素总数
     *     sum(hist[k], k=0..255) = totalPixels
     */
    int grayLevels = 256;
    vector<int> hist(grayLevels, 0);  // vector<int> = 动态数组，等价于 int hist[256]
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int val = img.at<uchar>(i, j);  // 取出像素 (i,j) 的灰度值（0~255）
            hist[val]++;                     // 投到对应桶中，计数 +1
        }
    }

    /*
     * 第二步：计算累计直方图，找到阈值 T
     *
     * 什么是累计直方图（Cumulative Histogram）？
     * cumHist[k] = hist[0] + hist[1] + ... + hist[k]
     *            = 灰度值 <= k 的像素总数
     *            = 在直方图上从 0 累加到 k 的结果
     *
     * 目标像素数 targetCount = p * totalPixels
     *   p = 0.5, totalPixels = 10000 -> targetCount = 5000
     *   意思是：我们需要找到灰度级 T，使得"比 T 暗的像素数" >= 5000
     *
     * 遍历直方图，找到第一个满足条件的灰度级 k：
     *   累加 hist[k] 到 cumulative
     *   如果 cumulative >= targetCount，说明 k 就是阈值 T
     *   为什么要找"第一个"满足条件的？
     *   因为我们是从灰度级 0 开始累加的，找到临界点就 break，
     *   这样保证"比 T 暗的像素数"刚好接近 p%，不多不少。
     *
     * 举例：
     *   hist[0]=100, hist[1]=200, hist[2]=300, ... targetCount=5000
     *   cumulative 从 100 -> 300 -> 600 -> ... 逐步累加
     *   当 cumulative >= 5000 时，假设当前 k=50，那么 T=50
     *   即：灰度值 <= 50 的像素数刚好达到了总像素的 p%
     */
    int targetCount = static_cast<int>(p * totalPixels);  // 目标像素数
    int cumulative = 0;      // 累计像素数，初始为 0
    int threshold = 0;        // 阈值，初始为 0
    for (int k = 0; k < grayLevels; k++)
    {
        cumulative += hist[k];  // 把当前灰度级的像素数加进来
        if (cumulative >= targetCount)  // 一旦达到目标，立即取阈值为 k
        {
            threshold = k;      // k 就是使累计像素数刚好达到 p% 的灰度级
            break;              // 找到了，不需要继续遍历
        }
    }

    /*
     * 第三步：用阈值 T 对图像做二值化
     *
     * 创建一张黑色的二值图像（所有像素初始化为 0 = 黑色）：
     *   Mat::zeros(height, width, CV_8UC1)
     *   CV_8UC1 = 8位无符号字符，1通道（灰度图）
     *   Scalar(0) = 所有像素初始值设为 0
     *
     * 然后遍历图像：
     *   如果像素灰度值 >= threshold → 设为 255（白色）
     *   如果像素灰度值 <  threshold → 保持 0（黑色）
     *
     * 为什么用 >= 而不是 >？
     *   边界情况：灰度值恰好等于 threshold 的像素，
     *   用 >= 会把它归入白色（前景），用 > 则归入黑色（背景）。
     *   两种方式都可以，>= 是更常见的选择。
     *
     * 注意这里的语义：
     *   灰度值 >= T（较亮）→ 白色 = 前景
     *   灰度值 <  T（较暗）→ 黑色 = 背景
     *   这意味着 p% 控制的是"较亮部分"的占比。
     */
    ThresholdPImg = Mat::zeros(height, width, CV_8UC1);  // 先创建全黑图像
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (img.at<uchar>(i, j) >= threshold)
                ThresholdPImg.at<uchar>(i, j) = 255;  // 较亮 → 白色
        }
    }

    return ThresholdPImg;  // 返回二值化结果
}


/*
 * 主函数：程序入口
 */
void main()
{
    /*
     * 读取 testimg.jpg 图像文件。
     * imread 以 BGR 格式读取。
     */
    Mat input = imread("testimg.jpg");

    /*
     * 将彩色图转换为灰度图。
     * 阈值化操作在灰度图上进行（每个像素只需一个值来判断 >= T 或 < T）。
     */
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);

    /*
     * 调用 p 率阈值化函数，对灰度图进行二值化。
     * 默认 p=0.5，即把图像分成"较亮的 50%"和"较暗的 50%"两部分。
     */
    Mat ThresholdPImg = myThresholdP(gray);

    /*
     * 显示三张图：
     *   input：       原始彩色图像
     *   gray：        灰度图
     *   ThresholdPImg：p 率阈值化后的二值图像
     *
     * 观察二值图像可以看到：
     *   较亮的部分（>=T）变成白色，较暗的部分（<T）变成黑色。
     */
    imshow("input", input);
    imshow("gray", gray);
    imshow("ThresholdPImg", ThresholdPImg);

    /*
     * 等待用户按键，参数 0 表示无限等待。
     */
    waitKey(0);
}
