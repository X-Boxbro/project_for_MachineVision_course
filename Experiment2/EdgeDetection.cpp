#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <cmath>

using namespace cv;
using namespace std;

/*
 * ================================================================
 * 边缘检测（Edge Detection）是什么？
 * ================================================================
 * 边缘，就是图像中亮度（灰度值）发生剧烈变化的地方。
 * 比如：一块黑色区域和白色区域交界的地方，就是一条"边缘"。
 * 边缘包含了图像中最重要的结构信息（比如物体的轮廓），
 * 所以检测边缘是机器视觉中最基础、最常用的操作之一。
 *
 * 本实验使用"Sobel 算子"来检测边缘，这是一种经典的"差分滤波器"。
 *
 * 什么是"差分"？
 * 你可以把图像想象成一张高低起伏的地形图（灰度值 = 高度）。
 * "差分"就是计算"相邻像素之间的差值"——
 * 如果相邻像素的灰度值差得很大，说明这里很可能是一条边缘。
 * 如果相邻像素的灰度值差不多，说明这里没有边缘。
 */

/*
 * myEdgeDetect: 使用 Sobel 差分滤波器检测图像边缘
 *
 * 核心思想：
 *   我们分别计算两个方向的梯度（差分）：
 *     - Gx（水平梯度）：检测"垂直方向"的边缘（左到右的灰度变化）
 *     - Gy（垂直梯度）：检测"水平方向"的边缘（上到下的灰度变化）
 *   然后把两个方向的结果合并，得到最终的边缘强度。
 *
 * 参数: img - 输入一张灰度图（每像素1个字节，0=黑，255=白）
 * 返回: 边缘图（同样大小的灰度图，值越大 = 边缘越明显）
 */
Mat myEdgeDetect(Mat img)
{
    /*
     * 第一步：定义 Sobel 卷积核（也叫"滤波器模板"）
     *
     * 一个卷积核就是一个 3x3 的小矩阵，里面每个位置有一个数字（权重）。
     * 我们用它和图像的每个 3x3 邻域做"对应位置相乘再相加"的运算。
     * 这个操作叫做"卷积"，是图像处理中最基础的操作。
     *
     * 为什么要定义两个核？
     * 因为单一方向的核只能检测一个方向的边缘，
     * 必须同时检测水平（Gx）和垂直（Gy）两个方向才能捕捉所有边缘。
     *
     * Gx 核（水平梯度核）：检测左到右的灰度变化
     *   -1  0  +1
     *   -2  0  +2    -> 左边的像素乘 -1（变暗权重），右边的像素乘 +1（变亮权重）
     *   -1  0  +1      中心行权重更大（-2 和 +2），所以水平方向的变化更敏感
     *
     * Gy 核（垂直梯度核）：检测上到下的灰度变化
     *   -1  -2  -1
     *    0   0   0   -> 上边的像素乘 -1（变暗权重），下边的像素乘 +1（变亮权重）
     *   +1  +2  +1     中心列权重更大（-2 和 +2），所以垂直方向的变化更敏感
     */
    Mat Gx_kernel = (Mat_<float>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);

    Mat Gy_kernel = (Mat_<float>(3, 3) <<
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1);

    /*
     * 卷积核的尺寸：3x3
     * kHalfH / kHalfW：核的半径（对于 3x3 的核，半径 = 1）
     * 这个半径在后面计算"当前像素的邻域坐标"时很重要：
     * 比如要处理第 (i, j) 个像素，它对应的 3x3 邻域覆盖了
     * 从 (i-1, j-1) 到 (i+1, j+1) 的区域。
     */
    int kh = 3;
    int kw = 3;
    int kHalfH = kh / 2;  // = 1，核在垂直方向的"半身高"
    int kHalfW = kw / 2;  // = 1，核在水平方向的"半身高宽"

    /*
     * 创建两个浮点型图像 Gx 和 Gy，用来存储每个像素的梯度值。
     * 为什么用 float（32位浮点数）而不是 uchar（8位整数）？
     * 因为卷积计算中，9个数相乘再相加，结果可能超过 255（uchar 的上限）。
     * 如果用 uchar，数据会"溢出"（wrap around），结果就完全错了。
     * 而 float 可以表示很大或很小的数，不会溢出。
     * CV_32FC1 表示：32位浮点数，1个通道（灰度图）
     */
    Mat Gx(img.rows, img.cols, CV_32FC1, Scalar(0));
    Mat Gy(img.rows, img.cols, CV_32FC1, Scalar(0));

    /*
     * 第二步：用 Gx 核做卷积，计算每个像素的水平梯度 Gx
     *
     * Gx > 0 表示：从左到右灰度值在增大（变亮）
     * Gx < 0 表示：从左到右灰度值在减小（变暗）
     * |Gx| 越大 = 垂直边缘越明显
     *
     * 实现细节：手动四重循环
     *   - 外两层循环：遍历图像中的每个像素 (i, j)
     *   - 内两层循环：遍历 3x3 卷积核的每个权重
     *     将核权重与对应邻域像素值相乘，然后累加
     */
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float sumX = 0.0f;
            for (int ki = 0; ki < kh; ki++) {
                for (int kj = 0; kj < kw; kj++) {
                    /*
                     * 计算核上第 (ki, kj) 个位置对应的邻域像素坐标
                     * 以图像像素 (i, j) 为中心，ki 和 kj 的偏移量决定了
                     * 取原图中哪个位置的像素与核当前位置的权重相乘。
                     * 举例：当 ki=0, kj=0 时，取的是 (i-1, j-1) 位置的像素
                     *       对应核的左上角 -1 这个权重
                     */
                    int row = i + ki - kHalfH;
                    int col = j + kj - kHalfW;
                    /*
                     * 边界处理（replicate 方式）：
                     * 当 row 或 col 超出图像范围时（比如计算边缘像素时），
                     * 不抛弃这些像素，而是把越界的坐标"夹取"到最近的边界像素。
                     * 效果就是：边缘像素会把边界外的"不存在"的像素
                     *          视为与边界像素相同的值。
                     * 具体实现：min(max(row, 0), img.rows-1)
                     *   如果 row < 0，取 0（第一行）
                     *   如果 row >= img.rows，取 img.rows-1（最后一行）
                     *   否则保持 row 不变
                     */
                    row = max(0, min(row, img.rows - 1));
                    col = max(0, min(col, img.cols - 1));
                    /*
                     * 取出邻域像素值（uchar 类型，范围 0~255），
                     * 乘以对应核权重（float 类型），
                     * 累加到 sumX 中。
                     */
                    sumX += img.at<uchar>(row, col) * Gx_kernel.at<float>(ki, kj);
                }
            }
            /*
             * 循环结束后，sumX 就是像素 (i, j) 的水平梯度 Gx。
             * 把它存到结果图像 Gx 中。
             */
            Gx.at<float>(i, j) = sumX;
        }
    }

    /*
     * 第三步：用 Gy 核做卷积，计算每个像素的垂直梯度 Gy
     *
     * Gy > 0 表示：从上到下灰度值在增大（变亮）
     * Gy < 0 表示：从上到下灰度值在减小（变暗）
     * |Gy| 越大 = 水平边缘越明显
     *
     * 代码逻辑完全同上，只是核换成了 Gy_kernel。
     */
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float sumY = 0.0f;
            for (int ki = 0; ki < kh; ki++) {
                for (int kj = 0; kj < kw; kj++) {
                    int row = i + ki - kHalfH;
                    int col = j + kj - kHalfW;
                    row = max(0, min(row, img.rows - 1));
                    col = max(0, min(col, img.cols - 1));
                    sumY += img.at<uchar>(row, col) * Gy_kernel.at<float>(ki, kj);
                }
            }
            Gy.at<float>(i, j) = sumY;
        }
    }

    /*
     * 第四步：合并 Gx 和 Gy，得到最终的边缘强度（梯度幅值）
     *
     * 对于每个像素 (i, j)，我们有：
     *   Gx = 该像素水平方向的梯度（检测垂直边缘）
     *   Gy = 该像素垂直方向的梯度（检测水平边缘）
     *
     * 为什么要用 sqrt(Gx^2 + Gy^2)？
     * 梯度本质上是一个向量（有大小和方向）。
     * Gx 和 Gy 分别是这个向量在 x 方向和 y 方向上的分量。
     * 根据向量叠加原理，总梯度大小 = sqrt(Gx^2 + Gy^2)。
     * 这个公式叫"二范数"或"欧几里得范数"。
     *
     * 举例：
     *   如果只有水平边缘 -> Gx != 0, Gy ~= 0 -> G ~= |Gx|
     *   如果只有垂直边缘 -> Gx ~= 0, Gy != 0 -> G ~= |Gy|
     *   如果是45度边缘 -> Gx = Gy -> G = sqrt(2) * |Gx|
     *
     * 先遍历一遍找出最大值，用于后面的"归一化"：
     */
    float maxVal = 0.0f;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float gx = Gx.at<float>(i, j);
            float gy = Gy.at<float>(i, j);
            float mag = sqrt(gx * gx + gy * gy);
            if (mag > maxVal) maxVal = mag;
        }
    }

    /*
     * 第五步：将梯度幅值归一化到 [0, 255] 范围
     *
     * 归一化 = 把数据缩放到一个标准范围。
     * 这里把每个像素的梯度幅值除以 maxVal，再乘以 255，
     * 使得最大的边缘响应变成 255（白色），其余按比例缩放。
     *
     * 为什么需要归一化？
     * 因为计算出的梯度值范围是不确定的（取决于图像内容），
     * 我们需要把它们映射到图像的灰度范围 [0, 255] 才能显示。
     *
     * saturate_cast<uchar>(...) 的作用：
     * 如果计算结果超出了 uchar 的范围（0~255），
     * 自动把它们截断到边界值（比如 -10 变成 0，300 变成 255），
     * 防止出现"数据溢出"的错误。
     */
    Mat EdgeImg(img.rows, img.cols, CV_8UC1, Scalar(0));
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float gx = Gx.at<float>(i, j);
            float gy = Gy.at<float>(i, j);
            float mag = sqrt(gx * gx + gy * gy);
            EdgeImg.at<uchar>(i, j) = saturate_cast<uchar>(mag / maxVal * 255.0f);
        }
    }

    /*
     * 返回边缘检测结果图。
     * 图像中，白色（值大）的地方代表检测到的边缘，
     * 黑色（值小）的地方代表没有边缘的区域。
     */
    return EdgeImg;
}


/*
 * 主函数：程序入口
 */
int main()
{
    /*
     * 读取一张图片 testimg.jpg。
     * imread 会把图像加载为 Mat 对象（OpenCV 中的图像容器）。
     * 读取后的图像默认是 BGR 彩色格式（注意不是常见的 RGB）。
     */
    Mat input = imread("testimg.jpg");

    /*
     * 将彩色图转换为灰度图。
     *
     * 为什么需要灰度图？
     * 彩色图像有 R、G、B 三个通道，处理起来更复杂。
     * 对于边缘检测、阈值分割等操作，我们只需要亮度信息就够了，
     * 所以通常先转成灰度图（只保留亮度，丢弃颜色信息）。
     *
     * cvtColor 是一个颜色空间转换函数。
     * COLOR_BGR2GRAY：BGR -> 灰度
     * 转换公式（简化版）：灰度 = 0.114*B + 0.587*G + 0.299*R
     * （人眼对绿色最敏感，所以绿色通道权重最大）
     */
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);

    /*
     * 调用我们前面实现的 myEdgeDetect 函数，
     * 对灰度图进行边缘检测，得到边缘图 EdgeImg。
     */
    Mat EdgeImg = myEdgeDetect(gray);

    /*
     * 显示图像
     * imshow("窗口标题", 图像对象)
     * 这里会弹出两个窗口：
     *   - "input"：显示原始彩色输入图像
     *   - "EdgeImg"：显示边缘检测结果
     */
    imshow("input", input);
    imshow("EdgeImg", EdgeImg);

    /*
     * 等待用户按键。
     * 参数 0 表示：无限等待，直到用户按任意键才继续执行。
     * 如果不加这一行，窗口会在创建后立即关闭，用户根本看不到图像。
     */
    waitKey(0);
}
