# Use the traied model to separate images & notes 
import os
import shutil
import PIL
from PIL import Image
from predict import *

"""
 The following code can be used to separate the images and notes .
 For example : If your initial folder looks like this(both notes and imgs in random order):
    myfolder:
            -notes1.jpg
            -img1.jpg
            -notes2.jpg
            -notes3.jpg
            -img2.jpg
            -...
    The final folder will be like:
    myfolder:
        notes:
            -notes1.jpg
            -notes2.jpg
            -notes3.jpg
        images:
            -img1.jpg
            -img2.jpg
USE ALEXNET FOR BETTER RESULTS (based on test results)
 """


 pathToMyfolder =  '**********'   #path to myfolder (which contains images & notes )
 os.makedir(os.path.join(pathToMyfolder,'notes'))
 os.makedir(os.path.join(pathToMyfolder,'images'))
 trained_model = torch.load('****') # path to saved model Or you can train a model using trained_model = train(dir,num_epochs,learning_rate,model_name) (train.py)
 src_files = os.listdir(pathToMyfolder)


 for file_name in src_files:
     img = Image.open(pathToMyfolder+'/'+file_name).convert('RGB')
     if(predict(img)==0):
         shutil.move(pathToMyfolder+'/'+file_name,pathToMyfolder+'/notes/'+file_name)       # class 0 corresponds to notes
     else:
         shutil.move(pathToMyfolder+'/'+file_name,pathToMyfolder+'/notes/'+file_name)
