# Project Name
Neural Networks Project for Gesture recognition

## General Information
Problem Statement:
Problem statement for the Neural Networks Project-Gesture recognition activity is as below:
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognize five different gestures performed by the user which will help users control the TV without using a remote.
The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:
•	Thumbs up:  Increase the volume
•	Thumbs down: Decrease the volume
•	Left swipe: 'Jump' backwards 10 seconds
•	Right swipe: 'Jump' forward 10 seconds  
•	Stop: Pause the movie

Dataset:
The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames (images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use. Videos have two types of dimensions - either 360x360 or 120x160. Each row of the CSV file represents one video and contains three main pieces of information - the name of the subfolder containing the 30 images of the video, the name of the gesture and the numeric label (between 0-4) of the video.

Activity Flow:
Train different models on the 'train' folder to identify the correct command in each sequence of frames and which also performs well on the 'val' folder. Final model's performance will be tested on the 'test' set which has been withheld for evaluation purpose.

Two Architectures for Video data analysis: 3D Convs and CNN-RNN Stack:
For analysing videos using neural networks, below two types of architectures are used commonly. 
a)	CNN + RNN
The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular Softmax  (for a classification problem such as this one).


b)	3D Convolutional Network, or Conv3D
3D convolutions are a natural extension to the 2D convolutions. Just like in 2D conv, you move the filter in two directions (x and y), in 3D Conv, you move the filter in three directions (x, y and z). In this case, the input to a 3D Conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels. Hence, deriving the analogy from 2-D convolutions where a 2-D kernel/filter (a square filter) is represented as (fxf)xc where f is filter size and c is the number of channels, a 3-D kernel/filter (a 'cubic' filter) is represented as (fxfxf)xc (here c = 3 since the input images have three channels). This cubic filter will now '3D-convolve' on each of the three channels of the (100x100x30) tensor.

## DL Framework used
Tensorflow
