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

# Creating the network
net = Net()

# Loading the saved network parameters
net = torch.load('./pretrained_models/run7/Network_7.pth')

# Checking if there is a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device being used =', device)

# Transferring the network onto the GPU
net.to(device)

# Ensuring that the model is in the training mode
net.eval()

# Stores the loss through out the entire training
testing_loss = []

# Hyper-Parameters
batch_size = 20

# Creating the data loader
test_dataset = dataset_pipeline(csv_file='/home/ecbm6040/dataset_final/test_2.csv', root_dir='/home/ecbm6040/dataset_update/test/')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# To compute the accuracy
num_correct = 0.0

print("------------------------------------------------------------------")

for i, batch in enumerate(test_dataloader):
	# Moving the mini-batch onto the GPU
	image, y = batch['image'].to(device), batch['labels'].to(device)
	y = y.resize((y.shape[0]))
	# print(y.shape)
	# print(image.shape)
	

	# Forward Propogation
	output = net(image)

	prediction = output.argmax(dim = 1).reshape((-1))
	num_correct += torch.sum(prediction == y) 

	print('prediction:', prediction)
	print('label:', y)

	print('num_correct:', torch.sum(prediction == y).item())

print("----------------------------------------------------------------------")
print("Testing Accuracy: {}".format(100 * num_correct.item() / 13888.0))
print("----------------------------------------------------------------------")















