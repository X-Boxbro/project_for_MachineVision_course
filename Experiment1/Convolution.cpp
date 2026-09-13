#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

Mat myConv(Mat img)
{
	Mat Conv_img;

	/*
	����ͼ����������ļ������
	*/

//����ֱ�ӷ���FFt+�����������ְ취ʵ�־���

	//����matlab�Ĵ���ʾ��������˹��
	int kernel_size = 128;
	double sigma = 20.0;
	Mat h = Mat::zeros(kernel_size, kernel_size, CV_64F);
	double sum = 0.0;
	int center = kernel_size / 2;

	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++) {
			double x = i - center;
			double y = j - center;
			h.at<double>(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma));
			sum += h.at<double>(i, j);
		}
	}
	// ��һ����������ֵ������=1�����Ȳ��䣩
	if (sum != 0) h /= sum;


	//1.ֱ�ӽ��о������ֶ�ʵ��
	Mat Conv_img_directly(img.rows, img.cols, CV_64F, Scalar(0));
	int kh = kernel_size;
	int kw = kernel_size;
	int kHalfH = center;
	int kHalfW = center;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			double sum = 0.0;
			for (int ki = 0; ki < kh; ki++) {
				for (int kj = 0; kj < kw; kj++) {
					int row = i + ki - kHalfH;
					int col = j + kj - kHalfW;
					if (row < 0 || row >= img.rows || col < 0 || col >= img.cols) {
						continue; // BORDER_CONSTANT: outside = 0
					}
					sum += img.at<uchar>(row, col) * h.at<double>(ki, kj);
				}
			}
			Conv_img_directly.at<double>(i, j) = sum;
		}
	}

	// ��һ���� 0~255
	double minVal, maxVal;
	minMaxLoc(Conv_img_directly, &minVal, &maxVal);
	if (maxVal != minVal) {
		Conv_img_directly = (Conv_img_directly - minVal) / (maxVal - minVal) * 255.0;
	}
	Conv_img_directly.convertTo(Conv_img, CV_8U);


	//2.ͨ��FFT/DFTʵ�־���������ͼƬ����
	//�Ȱ�ͼ��͸�˹�˵�����С
	int dft_rows = getOptimalDFTSize(img.rows + kernel_size - 1);
	int dft_cols = getOptimalDFTSize(img.cols + kernel_size - 1);
	Mat img_padded;
	copyMakeBorder(img, img_padded, 0, dft_rows - img.rows, 0, dft_cols - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat h_padded;
	copyMakeBorder(h, h_padded, 0, dft_rows - h.rows, 0, dft_cols - h.cols, BORDER_CONSTANT, Scalar::all(0));

	//ת�ɸ������Ͳ�DFT+���
	Mat img_dft, h_dft;
	dft(img_padded, img_dft, DFT_COMPLEX_OUTPUT);
	dft(h_padded, h_dft, DFT_COMPLEX_OUTPUT);
	mulSpectrums(img_dft, h_dft, img_dft, 0);

	//��DFT�õ����+�ü�
	Mat result;
	dft(img_dft, result, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	Rect roi(center, center, img.cols, img.rows);
	Conv_img = result(roi);
	double minVal, maxVal;
	minMaxLoc(Conv_img, &minVal, &maxVal);
	Conv_img -= minVal;
	Conv_img /= (maxVal - minVal);
	Conv_img *= 255.0;
	Conv_img.convertTo(Conv_img, CV_8U);



	//���ؾ����������ͼ��
	return Conv_img;
}

void main()
{
	Mat input = imread("testimg.jpg");

	Mat gray;
	//��ɫͼתΪ�Ҷ�ͼ
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//ͼ���������������ʵ��
	Mat Conv_img = myConv(gray);

	imshow("Conv_img", Conv_img);
	waitKey(0);
}