import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np 
import os 
from utils.batch_loader import dataset_pipeline
from models.model1 import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Creating the network
net = Net()

# Loading the saved network parameters
net = torch.load('./pretrained_models/run2/Network_2.pth')

# Checking if there is a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device being used =', device)

# Transferring the network onto the GPU
net.to(device)

# Ensuring that the model is in the training mode
net.eval()

src_dir = os.listdir('/home/ecbm6040/data_kaggle/asl_resized/')

A = np.zeros([len(src_dir), 279, 249, 3])

for i in range(len(src_dir)):
	A[i, :, :, :] = plt.imread('/home/ecbm6040/data_kaggle/asl_resized/' + src_dir[i])

A = A.transpose((0, 3, 1, 2))
print(A.shape)

