# Train and save the model
from shallownet import *
from deepwnet import *
from transfer_learning_alexnet import *
from transfer_learning_resnet import *
from train import *
import torch
# Path to images (arranged in the format requried by ImageFolder in Pytorch)
dir = '/content/drive/My Drive/ML/Datasets/imagesvsnotes'
num_epochs = 20
learning_rate = 0.0001
model_name='alexnet'
trained_model = train(dir,num_epochs,learning_rate,model_name)
torch.save(trained_model,'mymodel') # can save this model for later use
