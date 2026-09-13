clc;clear all;close all;

% 读入图像
img = imread('testimg.jpg');

% 将读入的彩色图像转换为灰度图像
img = im2gray(img);

%计算积分图像，需编程实现
InteImg_result = MyInteImg(img);

figure;
imshow(InteImg_result);
title('InteImg-result');