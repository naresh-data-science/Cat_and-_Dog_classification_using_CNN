
# Cat & Dog Classification using CNN:

Convolutional Neural Network (CNN) is an algorithm taking an image as input then assigning weights and biases to all the aspects of an image and thus differentiates one from the other. Neural networks can be trained by using batches of images, each of them having a label to identify the real nature of the image (cat or dog here). A batch can contain few tenths to hundreds of images. For each and every image, the network prediction is compared with the corresponding existing label, and the distance between network prediction and the truth is evaluated for the whole batch. Then, the network parameters are modified to minimize the distance and thus the prediction capability of the network is increased. The training process continues for every batch similarly.

## Architecture of CNN for Cat & Dog Classification

It has three layers namely convolutional, pooling, and a fully connected layer. It is a class of neural networks and processes data having a grid-like topology. The convolution layer is the building block of CNN carrying the main responsibility for computation
![App Screenshot](https://assets.skyfilabs.com/images/blog/image-classifer-for-identifying-cats-dogs.webp)





## Installing Required Packages for Python 3.6

1. Numpy -> 1.14.4 [ Image is read and stored in a NumPy array ] 2. TensorFlow -> 1.8.0 [ Tensorflow is the backend for Keras ] 3. Keras -> 2.1.6 [ Keras is used for implementing the CNN 

## Import Libraries

Installing Required Packages for Python 3.6
1. Numpy -> 1.14.4 [ Image is read and stored in a NumPy array ] 
2. TensorFlow -> 1.8.0 [ Tensorflow is the backend for Keras ] 
3. Keras -> 2.1.6 [ Keras is used for implementing the CNN

CNN does the processing of Images with the help of matrixes of weights known as filters. They detect low-level features like vertical and horizontal edges etc. Through each layer, the filters recognize high-level features.

We first initialize the CNN,

For compiling the CNN, we are using adam optimizer.

Adaptive Moment Estimation (Adam) is a method used for computing individual learning rates for each parameter. For loss function, we are using Binary cross-entropy to compare the class output to each of the predicted probabilities. Then it calculates the penalization score based on the total distance from the expected value.

Image augmentation is a method of applying different kinds of transformation to original images resulting in multiple transformed copies of the same image. The images are different from each other in certain aspects because of shifting, rotating, flipping techniques. So, we are using the Keras ImageDataGenerator class to augment our images.

We need a way to turn our images into batches of data arrays in memory so that they can be fed to the network during training. ImageDataGenerator can readily be used for this purpose. So, we import this class and create an instance of the generator. We are using Keras to retrieve images from the disk with the flow_from_directory method of the ImageDataGenerator class
## Convolution

Convolution is a linear operation involving the multiplication of weights with the input. The multiplication is performed between an array of input data and a 2D array of weights known as filter or kernel. The filter is always smaller than input data and the dot product is performed between input and filter array.

## Activation

The activation function is added to help ANN learn complex patterns in the data. The main need for activation function is to add non-linearity into the neural network.

## Pooling

The pooling operation provides spatial variance making the system capable of recognizing an object with some varied appearance. It involves adding a 2Dfilter over each channel of the feature map and thus summarise features lying in that region covered by the filter.

So, pooling basically helps reduce the number of parameters and computations present in the network. It progressively reduces the spatial size of the network and thus controls overfitting. There are two types of operations in this layer; Average pooling and Maximum pooling. Here, we are using max-pooling which according to its name will only take out the maximum from a pool. This is possible with the help of filters sliding through the input and at each stride, the maximum parameter will be taken out and the rest will be dropped.

The pooling layer does not modify the depth of the network unlike in the convolution layer.

## Fully Connected

The output from the final Pooling layer which is flattened is the input of the fully connected layer.

The Full Connection process practically works as follows:

The neurons present in the fully connected layer detect a certain feature and preserves its value then communicates the value to both the dog and cat classes who then check out the feature and decide if the feature is relevant to them.

## Full CNN overview:

We are fitting our model to the training set. It will take some time for this to finish.

We can predict new images with our model by predict_image function where we have to provide a path of new image as image path and using predict method. If the probability is more than 0.5 then the image will be of a dog else of cat.

## Features Provided:
· We can test our own images and verify the accuracy of the model.

· We can integrate the code directly into our other project and can be extended into a website or mobile application device.

· We can extend the project to different entities by just finding the suitable dataset, change the dataset and train the model accordingly.

## Conclusion:
I hope you now have a basic understanding of Convolutional Neural networks and can classify images of cats and dogs

contact we me:




https://www.linkedin.com/in/naresh-ch-835923226/
