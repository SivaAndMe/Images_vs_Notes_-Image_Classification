#Define resnet model
import torch
from torch import nn
from torchvision import models


def getResnetModel():
    # Using resnet18 considering memory & speed .However , other versions of resnet can also be used.
    ResnetModel = models.resnet18(pretrained=True,progress=True)

    # freeze weights
    for param in ResnetModel.parameters():
      param.requires_grad=False
    # alter the last layer to suit our purpose
    ResnetModel.fc = nn.Linear(ResnetModel.fc.in_features,1,bias=True)
    # fc is the name of last fc layer and use it infeatures & this new one(redifined fc) requires grad by default

    return ResnetModel
