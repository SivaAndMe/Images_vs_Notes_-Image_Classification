import torch
from torch import nn
from torchvision import models

def getAlexnetModel():
    AlexNet = models.alexnet(pretrained=True,progress=True)
    # freeze weights
    for param in AlexNet.parameters():
      param.requires_grad=False
    # alter the last layer to suit our purpose
    AlexNet.classifier[6] = nn.Linear(AlexNet.classifier[6].in_features,1,bias=True)#
    return AlexNet
