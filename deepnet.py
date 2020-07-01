import torch
from torch import nn

class ImagesvsNotesDeepNet(nn.Module):
  def __init__(self,in_features=in_features,hidden_features=hidden_features):
    super(ImagesvsNotesDeepNet,self).__init__()
    self.layer1 = nn.Linear(in_features,hidden_features,bias=True)
    self.layer2 = nn.Linear(hidden_features,1,bias=True)

  def forward(self,X):
    X = torch.flatten(X,start_dim=1)
    X = nn.Linear(100*100*3,50*50*3,bias=True)(X)
    X = nn.ReLU()(X)
    X = nn.Linear(50*50*3,25*25*3,bias=True)(X)
    X = nn.ReLU()(X)
    X = nn.Linear(25*25*3,1,bias=True)(X)
    return X
