#  calculate mean & std for transforms
import torch
from torch import nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader, Dataset,random_split
import numpy as np

#  These transforms are needed for calculating mean,std_for_custom_nets
trandforms = transforms.Compose([transforms.Scale((100,100)),transforms.ToTensor()])#,transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
dataset = datasets.ImageFolder(root=dir,transform=trandforms)
dataloader = DataLoader(dataset,batch_size=len(dataset))    # Use ALL THE IMAGES to calculate mean,std.dev (with batch_size as len(dataset))

# After the above transforms, all images are scaled to size of (100,100)
# and pixel values are standardized b/w 0 & 1(ToTensor() does this along with converting images to tensors)
xb,_ = next(iter(dataloader))
arr = xb.numpy()
# dummy means &stddevs (will be overwritten later)
mean = np.array([1,2,3],dtype=np.float64)#dtype is very important here . W/o this it's truncating floats to decimals
std = np.array([1,2,3],dtype=np.float64)
for i in range(3):
  brr = arr[:,i,:,:].flatten()
  mean[i] = np.mean(brr,dtype=np.float64) # across all the images and in a particular channel
  std[i] = np.std(brr,dtype=np.float64)
  # max is 1 and min is 0, no negs here !
print(mean)
print(std)
