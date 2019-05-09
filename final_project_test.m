% April 25, 2019

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

% Skin color values based on the paper
r_th = 95 / 255;
g_th = 40 / 255;
b_th = 20 / 255;

[m, n, k] = size(F); 

% Based on Sharma paper, did thresholding
f_th = (F(:,:,1) > r_th) .* (F(:,:,2) > g_th) .* (F(:,:,3) > b_th) .* ((max(F,[], 3) - min(F,[], 3)) > (15 / 255)) .* (abs(F(:,:,1) - F(:,:,2)) > 15/255) .* (F(:,:,1) > F(:,:,2)) .* (F(:,:,1) > F(:,:,3)); 
f_th_1 = (F(:,:,1) > r_th) .* (F(:,:,2) > g_th) .* (F(:,:,3) > b_th) .* ((max(F,[], 3) - min(F,[], 3)) <= (15 / 255)) .* (abs(F(:,:,1) - F(:,:,2)) > 15/255) .* (F(:,:,1) > F(:,:,2)) .* (F(:,:,1) > F(:,:,3)); 

se = [1 1; 1 1];
E = imerode(f_th,se);
E = imdilate(E,se);
E = imdilate(E,se);
E = imdilate(E,se);
E = imerode(E,se);

h = figure
subplot(1,4,1); imshow(F); title('Original Image')
subplot(1,4,2); imshow(f_th); title('>')
subplot(1,4,3); imshow(f_th_1);title('<=')
subplot(1,4,4); imshow(E); title('Erode, dilate thrice, erode')


saveas(h,sprintf('figure%d.png',i))
close all

end

%%
f1_gray = rgb2gray(F);
T = graythresh(f1_gray);
B = im2bw(f1_gray,T);

figure
imshow(B)




