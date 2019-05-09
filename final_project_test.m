% May 8, 2019
% This file uploads images and performs RGB and YCbCr thresholding, saving
% the corresponding figures.

%% Final Project April

close all;clear all;clc

f1 = imread('color_21_0146.png');
f2 = imread('color_20_0133.png');
f3 = imread('color_14_0101.png');
f4 = imread('color_14_0002.png');
f5 = imread('color_10_0421.png');
f6 = imread('color_0_0179.png');
f7 = imread('color_0_0048.png');
f8 = imread('color_0_0006.png');


%%
% Background subtraction - will not do because skin detection will do this
% anyway

%Thresholding
% This loop will automatically geenrate and save figures
for i=1:8

A = genvarname(strcat('f',num2str(i,'%d')));
F = im2double(eval(A));
YF = rgb2ycbcr(eval(A));

% Skin color values based on the paper
r_th = 95 / 255;
g_th = 40 / 255;
b_th = 20 / 255;

[m, n, k] = size(F); 

% Based on Sharma paper, did thresholding
f_th = (F(:,:,1) > r_th) .* (F(:,:,2) > g_th) .* (F(:,:,3) > b_th) .* ((max(F,[], 3) - min(F,[], 3)) > (15 / 255)) .* (abs(F(:,:,1) - F(:,:,2)) > 15/255) .* (F(:,:,1) > F(:,:,2)) .* (F(:,:,1) > F(:,:,3)); 
f_th_1 = (F(:,:,1) > r_th) .* (F(:,:,2) > g_th) .* (F(:,:,3) > b_th) .* ((max(F,[], 3) - min(F,[], 3)) <= (15 / 255)) .* (abs(F(:,:,1) - F(:,:,2)) > 15/255) .* (F(:,:,1) > F(:,:,2)) .* (F(:,:,1) > F(:,:,3)); 
YF_th = (YF(:,:,2) > 77).*(YF(:,:,2) < 127) .* (YF(:,:,3) > 133).*(YF(:,:,3) < 173);

h = figure
subplot(1,4,1); imshow(F); title('Original Image')
subplot(1,4,2); imshow(f_th); title('RGB Thresholding >')
%subplot(1,5,3); imshow(f_th_1);title('<=')
subplot(1,4,3); imshow(YF); title('YCbCr Image')
subplot(1,4,4); imshow(YF_th); title('YCbCr Thresholded')

saveas(h,sprintf('figure%d.png',i))
close all

end





