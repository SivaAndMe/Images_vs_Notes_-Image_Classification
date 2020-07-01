# Train the model
from shallownet import *
from deepwnet import *
from transfer_learning_alexnet import *
from transfer_learning_resnet import *
from data import *
import torch
from torch import nn
from torch.optim import Adam,SGD,RMSprop
def train(dir,num_epochs,learning_rate,model_name):

    if(model_name=='shallow'):
        model = ImagesvsNotesShallowNet()
        train_dataloader,val_dataloader = getDataLoaders(dir)
    elif(model_name=='deep'):
        model = ImagesvsNotesDeepNet()
        train_dataloader,val_dataloader = getDataLoaders(dir)
    elif(model_name=='conv'):
        model = ImagesvsNotesConvNet(5,1)
        train_dataloader,val_dataloader = getDataLoaders(dir)
    elif(model_name=='resnet'):
        model = getResnetModel()
        train_dataloader,val_dataloader = getDataLoadersResnet(dir)
    elif(model_name=='alexnet'):
        model = getAlexnetModel()
        train_dataloader,val_dataloader= getDataLoadersAlexnet(dir):
    else:
        raise Exception("Inappropriate model")



    opt = Adam(model.parameters(),lr=learning_rate)
    loss_func = nn.BCEWithLogitsLoss()
    # This loss function is similar to BCE with sigmoids, but this direct one offers numerical stability as per pytorchs docs


    for i in range(num_epochs):
      model.train()
      avgloss=0
      correct = 0
      for xb,yb in train_dataloader:
        pred = model(xb)
        yb = yb.reshape(yb.shape[0],1)
        yb = yb.type_as(pred)
        loss = loss_func(pred,yb)
        loss.backward() #calculate grads
        opt.step()#update params
        opt.zero_grad()#so that next time grad won't accumulate
        avgloss =avgloss+ loss.data
        pred = (pred>0.5).float()# bool to float
        correct = correct + (pred==yb).float().sum()
      print("Epoch {}/{} \n Training: Avgloss: {} Accuracy:{} ".format(i+1,num_epochs,avgloss,correct/len(train_dataset)))

      #validating
      with torch.no_grad():# for saving memory dont compute grads while testing
      # To set batchnorm/dropout in eval mode, and other things you sholdnt do while testing
        model.eval()
        dliter = iter(val_dataloader)
        xv,yv = dliter.next()
        predv = model(xv)
        predv = (predv>0.5)
        yv = yv.reshape(shape=(yv.shape[0],1))
        correc = (predv==yv).int().sum()

        print(" Validation: Correct:{}  Accuracy : {}".format(correc,(1.0*correc)/len(val_dataset)))
    return model# return trained model
