#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

Mat myHist(Mat img)
{
	Mat Hist;

	/*
	���ƻ���ֱ��ͼ�ļ������
	*/

	// 1. �ֶ�����Ҷ�ֱ��ͼ����ʹ�� calcHist��
	int histSize = 256;
	int height = img.rows;
	int width = img.cols;
	vector<int> hist(histSize, 0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int val = img.at<uchar>(i, j);
			hist[val]++;
		}
	}

	// 2. ��һ��ֱ��ͼ�Ա����
	int histHeight = 400;
	int binWidth = 2;
	int maxCount = 0;
	for (int i = 0; i < histSize; i++) {
		if (hist[i] > maxCount) maxCount = hist[i];
	}

	// 3. ��������
	Hist = Mat::zeros(histHeight, histSize * binWidth, CV_8UC3);

	// 4. �ֶ�����ֱ��ͼ��״ͼ����ʹ�� calcHist �� normalize��
	for (int i = 0; i < histSize; i++)
	{
		// ���߰�����ӳ�䵽 histHeight
		int barHeight = 0;
		if (maxCount > 0) {
			barHeight = static_cast<int>(static_cast<double>(hist[i]) / maxCount * histHeight * 0.9);
		}
		rectangle(Hist,
			Point(i * binWidth, histHeight - barHeight),
			Point((i + 1) * binWidth - 1, histHeight - 1),
			Scalar(200, 200, 200),
			FILLED);
	}

	//���ظ���ԭͼ����Ƶ�ֱ��ͼ
	return Hist;
}

void main()
{
	Mat input = imread("testimg.jpg");

	Mat gray;
	//��ɫͼתΪ�Ҷ�ͼ
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//ֱ��ͼ���ƣ�����ʵ��
	Mat Hist = myHist(gray);

	imshow("Hist", Hist);
	waitKey(0);
}