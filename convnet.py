#  ConvNet
import torch
from torch import nn

class ImagesvsNotesConvNet(nn.Module):
  def __init__(self,filter_size,stride):
    super(ImagesvsNotesConvNet,self).__init__()
    self.f = filter_size
    self.s = stride
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout2d(p=0.2,inplace=False)
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(self.f,self.f),stride=self.s,padding=0)
    self.maxp1 = nn.MaxPool2d(kernel_size=(self.f,self.f),stride=self.s,padding=0)
    self.conv2 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(self.f,self.f),stride=self.s,padding=0)
    self.maxp2 = nn.MaxPool2d(kernel_size=(self.f,self.f),stride=self.s,padding=0)
    self.fc1 = nn.Linear(84*84*3,1,bias=True)    #40*40*3
    # self.fc2 = nn.Linear(40*40*3,1,bias=True)

  def forward(self,X):
    X = self.conv1(X)
    X = self.relu(X)
    X = self.maxp1(X)
    # X = self.dropout(X)
    X = self.conv2(X)
    X = self.relu(X)
    X = self.maxp2(X)
    # X = self.dropout(X)
    X=torch.flatten(X,start_dim=1)
    X = self.fc1(X)
    # X = self.fc2(X)
    return X
