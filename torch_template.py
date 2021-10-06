from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import*
from datetime import datetime

def train(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))



def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction = 'sum').item()
			pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss/=len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def save_models(model):
    print()
    torch.save(model.state_dict(), "models/trained.model")
    print("****----Checkpoint Saved----****")
    print()

def main():
	#Load Datas
	data_transform = transforms.Compose([
		transforms.Resize((64,64)),
		transforms.ToTensor()
	])
	trainset = datasets.ImageFolder('dataset/train', transform = data_transform)
	testset = datasets.ImageFolder('dataset/test', transform = data_transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 8)
	testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True, num_workers = 8)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#Create Models and Stuffs
	model = NeuralNetwork().to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 1, gamma = 0.8)

	#Load Model If Available Minimal 1x training
	#model.load_state_dict(torch.load('models/trained.model'))
	
	# set your own epoch

	for epoch in range(50):
		train(model, device, trainloader, optimizer, epoch)
		scheduler.step()
		save_models(model)
		test(model, device, testloader)
	
	#For Testing Only
	
if __name__ == "__main__":
	main()
