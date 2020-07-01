# ShallowNet    
import torch
from torch import nn
in_features = 100*100*3
hidden_features = 50*50*3

class ImagesvsNotesShallowNet(nn.Module):
  def __init__(self,in_features=in_features,hidden_features=hidden_features):
    super(ImagesvsNotesShallowNet,self).__init__()
    self.layer1 = nn.Linear(in_features,hidden_features,bias=True)
    self.layer2 = nn.Linear(hidden_features,1,bias=True)

  def forward(self,X):
    X = torch.flatten(X,start_dim=1)#start flattening after batch-dimension
    X = self.layer1(X)
    X = nn.ReLU()(X)
    X = self.layer2(X)
    return X
