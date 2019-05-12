#May/11/2019
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import glob
import os
from tqdm import tqdm

def thresh_1(R,G, B, r_th, g_th, b_th):
    R1 = R[:] > r_th
    G1 = G[:] > g_th
    B1 = B[:] > b_th
    
    C1 = np.multiply(R1,G1)
    C2 = np.multiply(C1,B1)
    D = C2.astype(int)
    return D

def thresh_2(F, th):
    A1 = F.max(2)
    A2 = F.min(2)
    B = A1-A2
    C = B[:] > th
    return C

def thresh_3(F, th):
    A1 = np.absolute(F[:,:,0])
    A2 = F[:,:,1]
    B = A1-A2
    C = B[:] > th
    return C

def thresh_4(F):
    A = F[:,:,0] > F[:,:,1]
    B = A.astype(int)
    return B

def thresh_5(F):
    A = F[:,:,0] > F[:,:,2]
    B = A.astype(int)
    return B

# Skin color values based on the Sharma paper
r_th = 95 / 255;
g_th = 40 / 255;
b_th = 20 / 255;

path = "/home/ecbm6040/dataset_final/train/"
dirs = os.listdir(path)

for filename in tqdm(dirs): #glob.glob(path):
    #im1=Image.open(filename)
    im1 = plt.imread("/home/ecbm6040/dataset_final/train/"+filename)

    # Image shape
    m = im1.shape[0]; n = im1.shape[1]; k = im1.shape[2]
    #print(m, n, k)

    F1 = im1[:,:,0]
    F2 = im1[:,:,1]
    F3 = im1[:,:,2]

    D1 = thresh_1(F1,F2,F3,r_th,g_th,b_th)
    D2 = thresh_2(im1, 15/255)
    D3 = thresh_3(im1, 15/255)
    D4 = thresh_4(im1)
    D5 = thresh_5(im1)

    #Perform multiplication
    E1 = np.multiply(D1,D2)
    E2 = np.multiply(E1,D3)
    E3 = np.multiply(E2,D4)
    E4 = np.multiply(E3,D5)

    fig1 = plt.gcf()
    plt.imshow(E4,cmap=plt.get_cmap('gray'), vmin=0, vmax=255)#cmap='Grey_r')
    plt.axis('off')
    plt.imsave('/home/ecbm6040/dataset_final/train_segmented/{}'.format(filename),E4,cmap=plt.get_cmap('gray'), vmin=0, vmax=255)#cmap='Greys_r')
    plt.tight_layout()
    plt.close()
