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
# Stores the accuracy through out the entire training
training_acc = []

train_dataset = dataset_pipeline(csv_file='/home/ecbm6040/dataset_update/train.csv', root_dir='/home/ecbm6040/dataset_update/train/')
val_dataset = dataset_pipeline(csv_file='/home/ecbm6040/dataset_update/val.csv', root_dir='/home/ecbm6040/dataset_update/val/')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# To compute the accuracy
num_correct = 0.0

# loop over the dataset multiple times
for epoch in range(num_epochs):
	print("------------------------------------------------------------------")

	# Ensuring that the model is in the training mode
	net.train()

	# Stores the loss for an entire mini-batch
	running_loss = 0.0
	
	# Loop over the entire training datasets
	for i, batch in enumerate(train_dataloader):

		# Zero the parameter gradients
		optimizer.zero_grad()
		 
		# Moving the mini-batch onto the GPU
		image, y = batch['image'].to(device), batch['labels'].to(device)
		# image, y = batch['image'], batch['labels']
		y = y.resize((y.shape[0]))
		
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

		prediction = output.argmax(dim = 1).reshape((y.shape[0]))
		num_correct += torch.sum(prediction == y) 

		if (i % 20 == 0 and i != 0):    # print every 20 mini-batches
			acc = num_correct.item() / (20.0 * batch_size)
			print('epoch: {}, mini_batch: {} loss: {}, acc: {}'.format(epoch, i, running_loss / 20, acc))
			training_loss.append(running_loss / 20)
			training_acc.append(acc)
			running_loss = 0.0
			num_correct = 0.0

		if (i % 500 == 0 and i != 0):
			net.eval()
			val_acc = []
			 
			num_correct_val = 0
			for j, val_batch in enumerate(val_dataloader):
				net.to(device)

				# Moving the mini-batch onto the GPU
				image, y = val_batch['image'].to(device), val_batch['labels'].to(device)

				# image, y = batch['image'], batch['labels']
				y = y.resize((batch_size))
				
				# Forward Propogation
				output = net(image)
				#print(j)
				prediction = output.argmax(dim = 1).reshape((-1))
				#print(y.shape,prediction.shape,(torch.sum(prediction == y)).item())
				num_correct_val += torch.sum(prediction == y)

			print("----------------------------------------------------------------------")
			print("Validation Accuracy: {}".format(100 * num_correct_val.item() / 5000.0))
			print("----------------------------------------------------------------------")
			val_acc.append(100 * num_correct_val.item() / 5000.0)
			net.train()

		# num_correct_val = 0
		# for j, val_batch in enumerate(val_dataloader):
		# 	net.to(device)

		# 	# Moving the mini-batch onto the GPU
		# 	image, y = val_batch['image'].to(device), val_batch['labels'].to(device)
			
		# 	# Forward Propogation
		# 	output = net(image)

		# 	prediction = output.argmax(dim = 1).reshape((-1))
		# 	num_correct_val += torch.sum(prediction == y)

		# print("----------------------------------------------------------------------")
		# print("Validation Accuracy: {}".format(100 * num_correct_val.item() / 5000.0))
		# print("----------------------------------------------------------------------")
		# val_acc.append(100 * num_correct_val.item() / 5000.0)
		# net.train()


	# Saving the model
	torch.save(net, './pretrained_models/run4/Network_4.pth')


	# net.eval()
	# val_acc = []
	 
	# num_correct_val = 0
	# for j, val_batch in enumerate(val_dataloader):
	# 	net.to(device)

	# 	# Moving the mini-batch onto the GPU
	# 	image, y = val_batch['image'].to(device), val_batch['labels'].to(device)
		
	# 	# Forward Propogation
	# 	output = net(image)

	# 	prediction = output.argmax(dim = 1).reshape((-1))
	# 	num_correct_val += torch.sum(prediction == y)

	# print("----------------------------------------------------------------------")
	# print("Validation Accuracy: {}".format(100 * num_correct_val.item() / 5000.0))
	# print("----------------------------------------------------------------------")
	# val_acc.append(100 * num_correct_val.item() / 5000.0)
	# net.train()

loss_file = open('./pretrained_models/run4/loss.txt', '+w') # open a file in write mode
for item in training_loss:    # iterate over the list items
	loss_file.write(str(item) + '\n') # write to the file
loss_file.close()   # close the file 

acc_file = open('./pretrained_models/run4/acc.txt', '+w') # open a file in write mode
for item in training_acc:    # iterate over the list items
	acc_file.write(str(item) + '\n') # write to the file
acc_file.close()   # close the file 

val_file = open('./pretrained_models/run4/val_acc.txt', '+w') # open a file in write mode
for item in val_acc:    # iterate over the list items
	val_file.write(str(item) + '\n') # write to the file
val_file.close()   # close the file 

			
print('Finished Training')

# Plotting the loss curve
# plt.plot(training_loss)
