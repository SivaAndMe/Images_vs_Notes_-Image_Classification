
![Images v/s Notes](/images/img1.png)

**Problem Statement**\
*The below sight(of Pictures of note book images(may be of class notes,question papers,assignment sheets,..) mixed with general images) is very common in the mobile phone gallery of college-going students. And after the semester gets completed, they no longer need these images and they prefer to delete them. But finding and deleting them individually among many other photos isn't easy.* ***This project attempts to solve this problem with the help of Deep Learning.*** . 

![Problem Statement](/images/img2.jpg)


* This project is helpful in classifying a given image as picture of notes or some other image. So it is a binary image classification problem.
Various classifiers are used to solve this problem - Standard shallow neural networks, Standard deep neural networks, Convolutional Neural Networks, pretrained neural networks(AlexNet and Resnet). One can directly use this code to separate notes images from others & delete them , if they want to.
The performance of these models on test data is in TestResults.txt (AlexNet=Resnet>ConvNet>ShallowNet>DeepNet). Ofcourse Alexnet takes just 5 epochs to achieve this performance while resnet needed 50 epochs.

* Pytorch framework is used.

* The high accuracies with test-data (as high as 99% with Transfer learning, 97% with ConvNet) can be attributed to the ease of classification between Notes and normal pictures.Further, these high performance metrics are helpful in building a good classifier to separate notes pictures from images.

* For standardizing the images, we can use ImageNet stats for mean and standard deviation for Alexnet & Resnet(which are trained on ImageNet). However, for our own standard networks(ShallowNet,DeepNet,ConvNets) we need to manually calculate mean and std.dev as in calculate_mean&std_for_custom_nets.py

* The training and validation results is in TrainResults.py .

* separate_files.py can be used to separate the images and notes . For example : If your initial folder looks like this(both notes and imgs in random order):
   * myfolder:
       * notes1.jpg
       * img1.jpg
     -   notes2.jpg
     -   notes3.jpg
     -   img2.jpg
     -    ..
* The final folder will be like:
   * myfolder:
       * notes:
           - notes1.jpg
           - notes2.jpg
           - notes3.jpg
       * images:
           - img1.jpg
           - img2.jpg
