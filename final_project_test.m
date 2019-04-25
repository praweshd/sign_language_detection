% April 25, 2019

%% Final Project April

close all;clear all;clc

f2 = imread('color_0_0002.png');
f1 = imread('color_17_0002.png');
f3 = imread('depth_0_0528.png');

%imshow(im2double(f3))

%%

% figure(1)
% subplot(1,2,1); imshow(f1)
% subplot(1,2,2); imshow(f2)
% 

% Background subtraction - will not do because skin detection will do this
% anyway

%Thresholding

%f1_gray = rgb2gray(f1);
f1 = im2double(f1);

r_th = 95 / 255;
g_th = 40 / 255;
b_th = 20 / 255;

[m, n, k] = size(f1); 

% Based on Sharma paper, did thresholding

f_th = (f1(:,:,1) > r_th) .* (f1(:,:,2) > g_th) .* (f1(:,:,3) > b_th) .* ((max(f1,[], 3) - min(f1,[], 3)) > (15 / 255)) .* (abs(f1(:,:,1) - f1(:,:,2)) > 15/255) .* (f1(:,:,1) > f1(:,:,2)) .* (f1(:,:,1) > f1(:,:,3)); 

f_th_1 = (f1(:,:,1) > r_th) .* (f1(:,:,2) > g_th) .* (f1(:,:,3) > b_th) .* ((max(f1,[], 3) - min(f1,[], 3)) <= (15 / 255)) .* (abs(f1(:,:,1) - f1(:,:,2)) > 15/255) .* (f1(:,:,1) > f1(:,:,2)) .* (f1(:,:,1) > f1(:,:,3)); 

figure(1)
subplot(1,3,1); imshow(f1);
subplot(1,3,2); imshow(f_th);
subplot(1,3,3); imshow(f_th_1);

figure(2)
%se = strel('ball',10,5);
% se = [1 1 1; 1 1 1; 1 1 1];
se = [1 1; 1 1];
E = imerode(f_th,se);
E = imdilate(E, se);
E = imdilate(E,se);
E = imdilate(E,se);
E = imerode(E,se);
imshow(E)

%%

f1_gray = rgb2gray(f1);
T = graythresh(f1_gray);
B = im2bw(f1_gray,T);

figure
imshow(B)




