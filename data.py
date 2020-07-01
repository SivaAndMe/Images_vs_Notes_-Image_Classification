# Define dataloaders
import torch
from torch import nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader, Dataset,random_split
torch.manual_seed(1)
bsize=32
# for ShallowNet,DeepNet,ConvNet
def getDataLoaders(dir):      #directory in which images are stored in the way ImageFolder in Pytorch requires.
    mean = [0.46807061, 0.46095277, 0.44188677] #These are the calculated stats using calculate_mean&std_for_custom_nets.py
    std =  [0.23720286, 0.22541129, 0.23456475]
    trandforms = transforms.Compose([transforms.Scale((100,100)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
    data = datasets.ImageFolder(root=dir,transform=trandforms)
    train_dataset,val_dataset = random_split(data,[int(0.84*len(data)),int(0.16*len(data))])#168,32
    train_dataloader = DataLoader(train_dataset,batch_size=bsize)
    val_dataloader = DataLoader(val_dataset,batch_size=bsize)#len(val_dataset)
    return train_dataloader,val_dataloader


# Redefining DataLoader for resnet & alexnet bcoz mean,std are to be imagenet stats and Scale varies with each network(i.e., different scale for alexnet,resnet..)
def getDataLoadersResnet(dir):
    mean = [0.485, 0.456, 0.406]    # Stats of ImageNet
    std =  [0.229, 0.224, 0.225]
    trandforms = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
    data = datasets.ImageFolder(root=dir,transform=trandforms)
    train_dataset,val_dataset = random_split(data,[int(0.84*len(data)),int(0.16*len(data))])#168,32
    train_dataloader = DataLoader(train_dataset,batch_size=bsize)
    val_dataloader = DataLoader(val_dataset,batch_size=bsize)
    return train_dataloader,val_dataloader

def getDataLoadersAlexnet(dir):
    mean = [0.485, 0.456, 0.406]    # Stats of ImageNet
    std =  [0.229, 0.224, 0.225]
    trandforms = transforms.Compose([transforms.Scale((256,256)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
    data = datasets.ImageFolder(root=dir,transform=trandforms)
    train_dataset,val_dataset = random_split(data,[int(0.84*len(data)),int(0.16*len(data))])#168,32
    train_dataloader = DataLoader(train_dataset,batch_size=bsize)
    val_dataloader = DataLoader(val_dataset,batch_size=bsize)
    return train_dataloader,val_dataloader
