# Predict
import torch
from torchvision import datasets,transforms,models


def predict(img,trained_model):
    # If the trained_model is not alexnet/resnetm, then you have to change these mean & std (ofcourse, dont have much problem if you don't change)
    mean = [0.485, 0.456, 0.406]    # Stats of ImageNet
    std =  [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
    img = transform(img)
    trained_model.eval()
    pred = trained_model(img)[0]
    pred = torch.round(pred)
    return (int)pred
