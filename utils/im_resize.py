import os 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = 279
N = 249

itr = 1

src_dir = os.listdir('/home/ecbm6040/dataset')
max_itr = len(src_dir)

for im_name in src_dir:
    template = np.zeros((M, N, 3))
    im_path = os.path.join('/home/ecbm6040/dataset', im_name)
    
    # Read the image
    im = plt.imread(im_path)
    # Get shape
    m, n, _ = im.shape

    # Padding the image
    template[(M-m)//2 : ((M-m)//2 + m), (N-n)//2 : ((N-n)//2 + n), :] = im

    # Save the image
    dst_path = os.path.join('/home/ecbm6040/dataset_final', im_name)
    plt.imsave(dst_path, template)

    print('{} / {}' .format(itr, max_itr))
    itr += 1