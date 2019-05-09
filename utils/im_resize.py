import os 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = 279
N = 249

src_dir = os.listdir('/home/ecbm6040/dataset/')

for im_name in src_dir:
	template = np.zeros((M, N, 3))

	# Read the image
	im = plt.imread(src_dir + im_name)

	# Get shape
	m, n, _ = im.shape

	# Padding the image
	template[np.floor((M-m)/2) : (np.floor((M-m)/2) + m), np.floor((N-n)/2) : (np.floor((N-n)/2) + n), :] = im

	# Save the image
	plt.imsave('/home/ecbm6040/dataset_final/' + im_name + '.png', template)

	break