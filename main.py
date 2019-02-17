import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# Load MNIST
batch_size=60
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


# Define block
class BasicBlock(nn.Module):
	def __init__(self, channel_num):
		super(BasicBlock, self).__init__()
		
		#TODO: 3x3 convolution -> relu
		#the input and output channel number is channel_num
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
		)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		
		#TODO: forward
		residual = x
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = x + residual
		out = self.relu(x)
		return out


# Define network
class Net100(nn.Module):
	def __init__(self):
		super(Net100, self).__init__()
		channel_num = 16
		
		#TODO: 1x1 convolution -> relu (to convert feature channel number)
		self.init_block = nn.Sequential(
			nn.Conv2d(1, channel_num, 1),
			nn.ReLU(),
		) 
		#TODO: stack 100 BasicBlocks
		self.basic_blocks = nn.ModuleList([BasicBlock(channel_num) for i in range(100)])

		#TODO: 1x1 convolution -> sigmoid (to convert feature channel number)
		self.final_block = nn.Sequential(
			nn.Conv2d(channel_num, 1, 1),
			nn.Sigmoid(),
		) 

	def forward(self, x):
		
		#TODO: forward
		x = self.init_block(x)
		for i, _ in enumerate(self.basic_blocks):
			x = self.basic_blocks[i](x)
		
		out = self.final_block(x)
		return out

# Use cuda
network = Net100().cuda()

# Optimizer
optimizer = optim.Adam(network.parameters(), lr = 0.001)

network.train()
time_start = time.time()
for epoch in range(1):
	for i, data in enumerate(trainloader, 0):
		img, label = data
		img = img.cuda()
		optimizer.zero_grad()

		# forward, backward, optimize
		if (i % 2 == 0):
			img = 1 - img
		recon = network(img)
		loss_net = torch.mean((recon-img) ** 2)
		#loss_net = torch.mean(torch.abs(recon-img))
		loss_net.backward()
		optimizer.step()

		# show results
		input = img.cpu().data[0,0,:,:]
		output = recon.cpu().data[0,0,:,:]
		plt.figure("input")
		plt.imshow(input,cmap="gray")
		plt.pause(0.001)
		plt.figure("output")
		plt.imshow(output,cmap="gray")
		plt.pause(0.001)
		print('[%d/%d, %d/%d] loss: %.5f, time: %.5f' % (epoch, 1, i, int(len(trainset)/batch_size), loss_net.data, time.time()-time_start))


