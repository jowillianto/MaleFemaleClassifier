import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from model import *
import numpy as np
import matplotlib.pyplot as plt

#Global Variables
Classes = ('Female', 'Male  ')

#Image Show Function 
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#Loading Neural Network Trained Data
def LoadNN(NeuralNetwork):
    NeuralNetwork.load_state_dict(torch.load('models/trained.model'))

#data tranformations 
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

#Data Loading
TestSet = datasets.ImageFolder('dataset/test', data_transform)
TestLoader = torch.utils.data.DataLoader(TestSet, batch_size = 20)
Data = iter(TestLoader)
Image, Target = Data.next()

#Testing with neural Network 
Net = NeuralNetwork()
LoadNN(Net)
with torch.no_grad():
    Output = Net(Image).argmax(dim = 1, keepdim = True)
    print(Output)

print('Truth     : ', ' '.join(Classes[Target[j]] for j in range(20)), ' ')
print('Predicted : ', ' '.join(Classes[Output[j]] for j in range (20)), ' ')
imshow(torchvision.utils.make_grid(Image))