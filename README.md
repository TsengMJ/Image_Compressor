# Image Compressor

## Introduction
  Recent years has seen an increased amount of research in image compression. Most of effort, however, has focused on how to use convolution neural network (CNN) to enhance image compression. One of them effort to use neural network to decrease the artifact of lossy image compression.  
  
  Ringing and blocking effect are two common, important, and unavoidable artifact of lossy image compression . Even though there are complex deblocking filters in recently compression techniques, for example H.264, there isn’t similar deblocking approach in wide used JPEG
  
  We will propose a convolution neural network which input is a decompressed image and output is a restored image to suppress ringing and blocking effect  and also have a satisfying performance.

## Network Architecture
Conv Unit which contains a convolutions layer with 3x3 kernel size followed by an 1x1 convolutions layer and a PReLU then followed by another 3x3 a convolutions layer and a PReLU. 

After three sequential Conv Unit, connect a 1x1 convolutions layer, 5x5 convolutions layer, and a PReLU.

![](/Result/Model.png)

### Loss Function
The mean-square error (MSE) is commonly used to compare image compression quality. We use MSE as our comparison standard and as our loss function, the formula is defined as:
MSE=1WHi(xi-xi)

Where *xi* and *x*i* are the values of the i-th pixel in *X* and *X**, *W* and *H* are the width and height of *X*. *X* denotes the original image and *X** denotes the restored image. 

## Dataset
The training dataset we used are **[Open Images 2019 (Google)](https://www.kaggle.com/c/open-images-2019-object-detection)**, **[BSR](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)**, and **[DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)**. The training data is more than 10 million,contain lots of kinds of situations, and the size of validation date is about 10 thousand 2k images. 

At the training phase we also augmented the images with rotation and resizing for trying to make the training more comprehensive.

## Result
![](/Result/1.png)
![](/Result/7.png)
![](/Result/4.png)
![](/Result/6.png)

