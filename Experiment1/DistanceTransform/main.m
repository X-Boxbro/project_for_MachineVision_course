clc;clear all;close all;

% 读入图像
img = imread('testimg.jpg');

% 将读入的彩色图像转换为二值图像
bw = im2bw(img);

%距离变换，需编程实现
DisTrans_result = MyDisTrans(bw);

figure;
imshow(DisTrans_result);
title('DisTrans-result');