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

# Checking if there is a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device being used =', device)

# Transferring the network onto the GPU
net.to(device)

# Ensuring that the model is in the training mode
net.train()

# Choosing the loss function criteria
criterion = nn.CrossEntropyLoss()    # This is the Cross Entropy Loss Function

# Choosing the optimizer and its hyper-parameters
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)    # Adaptive Momentum Optimizer

# Hyper-Parameters
num_epochs = 20
batch_size = 20

# Stores the loss through out the entire training
training_loss = []

train_dataset = dataset_pipeline(csv_file='/home/ecbm6040/dataset_final/train.csv', root_dir='/home/ecbm6040/dataset_final/train/')
val_dataset = dataset_pipeline(csv_file='/home/ecbm6040/dataset_final/val.csv', root_dir='/home/ecbm6040/dataset_final/val/')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# loop over the dataset multiple times
for epoch in range(num_epochs):  
	
	# Ensuring that the model is in the training mode
	net.train()

	# Stores the loss for an entire mini-batch
	running_loss = 0.0
	
	# Loop over the entire training dataset
	for i, batch in enumerate(train_dataloader):


		# Zero the parameter gradients
		optimizer.zero_grad()
		 
		# Moving the mini-batch onto the GPU
		image, y = batch['image'].cuda(), batch['labels'].cuda()
		# image, y = batch['image'], batch['labels']
		y = y.resize((batch_size))
		
		# Forward Propogation
		output = net(image)
		
		# Computng the loss
		loss = criterion(output, y)
		   
		# Back Propogation    
		loss.backward()
		
		# Updating the network parameters
		optimizer.step()

		# Print Loss
		running_loss += loss.item()

		if i % 20 == 0 and i != 0:    # print every 20 mini-batches
			print('epoch: {}, mini_batch: {} loss: {}'.format(epoch, i, running_loss / 20))
			training_loss.append(running_loss / 20)
			running_loss = 0.0

		break

	# Saving the model
	torch.save(net, 'Network_1.pth')
	
	net.eval()

	num_correct = 0
	for j, val_batch in enumerate(val_dataloader):
		net.to(device)

		# Moving the mini-batch onto the GPU
		image, y = val_batch['image'].to(device), val_batch['labels'].to(device)
		
		# Forward Propogation
		output = net(image)

		prediction = output.argmax(dim = 1).reshape((-1))

		num_correct += torch.sum(prediction == y) 
	
	acc = 100 * num_correct / 5000
	print('Validation accuracy after {} epochs is {}%'.format(epoch, acc))


			
print('Finished Training')

# Plotting the loss curve
plt.plot(training_loss)