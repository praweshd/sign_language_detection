import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining the network architecture
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.conv1 = nn.Conv2d(3, 64, 3)
		self.conv1_bn = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, 3)
		self.conv2_bn = nn.BatchNorm2d(64)

		self.conv3 = nn.Conv2d(64, 128, 3)
		self.conv3_bn = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(128, 128, 3)
		self.conv4_bn = nn.BatchNorm2d(128)

		self.conv5 = nn.Conv2d(128, 256, 3)
		self.conv5_bn = nn.BatchNorm2d(256)
		self.conv6 = nn.Conv2d(256, 256, 3)
		self.conv6_bn = nn.BatchNorm2d(256)

		self.conv7 = nn.Conv2d(256, 512, 3)
		self.conv7_bn = nn.BatchNorm2d(512)
		self.conv8 = nn.Conv2d(512, 512, 3)
		self.conv8_bn = nn.BatchNorm2d(512)

		# self.fc1 = nn.Linear(14*12*512, 4096)
		self.fc1 = nn.Linear(13*11*512, 4096)
		self.fc1_bn = nn.BatchNorm1d(4096)

		self.fc2 = nn.Linear(4096, 512)
		self.fc2_bn = nn.BatchNorm1d(512)

		self.fc3 = nn.Linear(512, 24)

		self.out_act = nn.Softmax()

		self.pool = nn.MaxPool2d(2, 2)

	def forward(self, x):
		x = F.relu(self.conv1_bn(self.conv1(x)))
		x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))

		x = F.relu(self.conv3_bn(self.conv3(x)))
		x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))

		x = F.relu(self.conv5_bn(self.conv5(x)))
		x = self.pool(F.relu(self.conv6_bn(self.conv6(x))))

		x = F.relu(self.conv7_bn(self.conv7(x)))
		x = self.pool(F.relu(self.conv8_bn(self.conv8(x))))

		# x = x.view(-1, 14*12*512)
		x = x.view(-1, 13*11*512)
		self.fc1 = nn.Linear(13*11*512, 4096)

		x = F.relu(self.fc1_bn(self.fc1(x)))
		x = F.relu(self.fc2_bn(self.fc2(x)))
		x = self.out_act(self.fc3(x))