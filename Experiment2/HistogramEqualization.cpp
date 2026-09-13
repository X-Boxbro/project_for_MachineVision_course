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
 * 直方图均衡化（Histogram Equalization）是什么？
 * ================================================================
 * 首先，我们需要理解"直方图"是什么。
 *
 * 直方图（Histogram）：
 *   统计图像中每个灰度值（0~255）出现的像素个数，
 *   横轴是灰度值（0=黑，255=白），纵轴是该灰度值的像素数量。
 *   直方图反映了图像的"明暗分布"情况。
 *
 * 举例：一张"雾蒙蒙"的图像
 *   大部分像素集中在灰度值 100~180 之间，
 *   两端（很黑和很白）的像素很少 → 图像看起来灰蒙蒙的，对比度低。
 *
 * 直方图均衡化要做什么？
 *   通过一个数学变换，把原本集中在中间范围的灰度值，
 *   分散到整个 0~255 的范围，让亮的地方更亮，暗的地方更暗。
 *   效果就是：图像的对比度（明暗差异）增强了，看起来更清晰。
 *
 * 核心原理：
 *   1. 先统计原图的直方图（每个灰度级有多少像素）
 *   2. 再统计"累计直方图"（CDF：Cumulative Distribution Function）
 *      CDF(k) = 所有灰度 <= k 的像素总数
 *   3. 用 CDF 构建一个"映射表"（LUT：Look-Up Table）
 *      新灰度 = round((L-1) / (M*N) * CDF(原灰度))
 *      其中 L=256（灰度级数），M*N = 图像总像素数
 *   4. 用这个映射表，把原图的每个像素换成新灰度
 *
 * 为什么叫"均衡化"？
 *   因为变换后，直方图会变得更"平坦"（均匀分布），
 *   各个灰度级的像素数量趋于相等。
 */

/*
 * myEqualizeHist: 手动实现直方图均衡化
 *
 * 参数: img - 输入灰度图（每像素1个字节，0~255）
 * 返回: 均衡化后的灰度图
 */
Mat myEqualizeHist(Mat img)
{
    Mat EqualizedImg = img.clone();  // 先复制一份原图，结果会覆盖在这个副本上
    int height = EqualizedImg.rows;  // 图像高度（像素行数）
    int width = EqualizedImg.cols;   // 图像宽度（每行像素数）
    int grayLevel = 256;            // 灰度级总数：0~255 共 256 个级别

    /*
     * Step 1: 统计直方图（Histogram / PDF：概率密度函数）
     *
     * 什么是直方图？
     * 直方图是一个长度为 256 的数组 hist[]，
     * hist[k] 表示：灰度值为 k 的像素，在整张图像中出现了多少次。
     *
     * 实现方法：遍历图像的每个像素，按照其灰度值"投票"到对应位置。
     * 比如某个像素的灰度值是 120，就执行 hist[120]++，表示"第120桶加1票"。
     *
     * 为什么 hist[256] 用 int 类型而不是 uchar？
     * 一张 1920x1080 的图像有超过 200 万个像素，
     * hist[某个值] 的累加结果可能达到数十万，超出 uchar 的上限 255。
     */
    int hist[256] = { 0 };
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // img.at<uchar>(i, j) 取出图像第 i 行第 j 列像素的灰度值（0~255）
            hist[EqualizedImg.at<uchar>(i, j)]++;
        }
    }

    /*
     * Step 2: 计算累计直方图（CDF：Cumulative Distribution Function）
     *
     * 什么是累计直方图？
     * cumHist[k] = 所有灰度值 <= k 的像素总数。
     * 也就是把前 k+1 个直方图桶累加起来。
     *
     * 举例：
     *   cumHist[0]  = hist[0]                                    （只有灰度0的像素）
     *   cumHist[1]  = hist[0] + hist[1]                          （灰度0和1的像素总数）
     *   cumHist[127] = hist[0] + hist[1] + ... + hist[127]       （灰度<=127的像素总数）
     *   cumHist[255] = 总像素数 M*N                               （所有像素）
     *
     * 累计直方图有什么用？
     * 它告诉了我们"灰度值 x 在原图中排在前百分之几"。
     * 这个比例就是均衡化映射的关键。
     */
    int cumHist[256] = { 0 };
    cumHist[0] = hist[0];  // 第一个灰度级的累计，就是它本身
    for (int k = 1; k < grayLevel; k++) {
        // 累加：第 k 灰度级的累计 = 第 k-1 灰度级的累计 + 第 k 灰度级本身的数量
        cumHist[k] = cumHist[k - 1] + hist[k];
    }

    /*
     * Step 3: 构建均衡化映射表（LUT：Look-Up Table）
     *
     * 什么是 LUT？
     * 就是一个"翻译表"：对于每个旧灰度值 k，告诉它应该变成什么新灰度值。
     * 我们不需要每次查表时都重新计算，直接查表效率更高。
     *
     * 均衡化公式：
     *   s_k = round((L - 1) / (M * N) * cumHist[k])
     *
     * 参数解释：
     *   s_k     = 灰度值 k 映射后的新灰度值
     *   L       = 灰度级总数 = 256
     *   M * N   = 图像总像素数
     *   cumHist[k] = 灰度值 <= k 的像素总数（累计直方图）
     *   round() = 四舍五入到最近的整数（因为灰度值必须是整数）
     *
     * 这个公式的含义：
     *   cumHist[k] / (M*N)  就是"灰度值 k 在原图中排在前百分之几"
     *   乘以 (L-1) = 255，就是把这个比例映射到 0~255 的范围
     *
     * 举例：
     *   假设 M*N = 10000，cumHist[100] = 5000
     *   s_100 = round(255 * 5000 / 10000) = round(127.5) = 128
     *   这表示：原图中灰度值为 100 的像素，现在变成 128（变亮了）
     *
     * saturate_cast<uchar>(...) 确保结果在 0~255 范围内。
     */
    int L = grayLevel;              // = 256
    int MN = height * width;         // 图像总像素数
    uchar map[256];                   // 映射表：map[k] = 均衡化后灰度值 k 应变成的值
    for (int k = 0; k < grayLevel; k++) {
        map[k] = saturate_cast<uchar>(round((double)(L - 1) * cumHist[k] / MN));
    }

    /*
     * Step 4: 应用映射表，对原图进行均衡化变换
     *
     * 遍历图像的每个像素，用映射表把旧灰度值替换成新灰度值。
     * 简单说就是：img[i][j] = map[img[i][j]]
     *
     * 这一步完成后，EqualizedImg 就是均衡化后的结果图了。
     */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            EqualizedImg.at<uchar>(i, j) = map[EqualizedImg.at<uchar>(i, j)];
        }
    }

    // 返回均衡化后的图像
    return EqualizedImg;
}


/*
 * 主函数：程序入口
 */
int main()
{
    /*
     * 读取 testimg.jpg 图像文件。
     * imread 默认以 BGR 格式读取（OpenCV 的内部格式）。
     */
    Mat input = imread("testimg.jpg");

    /*
     * 将彩色图转换为灰度图。
     * 直方图均衡化是针对灰度图像的操作，
     * 所以先把 3 通道彩色图转成单通道灰度图。
     * cvtColor 的 COLOR_BGR2GRAY 参数表示：BGR -> 灰度
     */
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);

    /*
     * 调用我们实现的 myEqualizeHist 函数，
     * 对灰度图进行直方图均衡化。
     */
    Mat EqualizedImg = myEqualizeHist(gray);

    /*
     * 显示原始灰度图和均衡化后的结果图。
     * 对比两者可以看到：
     *   - 原始图可能灰蒙蒙的，对比度低
     *   - 均衡化后，对比度增强，细节更清晰
     */
    imshow("Original Gray", gray);        // 窗口标题"Original Gray"显示原始灰度图
    imshow("Equalized Image", EqualizedImg);  // 窗口标题"Equalized Image"显示均衡化结果

    /*
     * 等待用户按键，参数 0 表示无限等待。
     * 按任意键后程序结束，窗口关闭。
     */
    waitKey(0);
    return 0;
}
