#**Traffic Sign Recognition** 


## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image01]: ./images/training_sample.png "Training Samples"
[image02]: ./images/training_histogram.png "Training Histogram"
[image03]: ./images/original_image.png "Original Image"
[image04]: ./images/preprocessed_image.png "Preprocessed Image with Grayscalling"
[image05]: ./images/example_00001.png "Example Traffic Sign"
[image06]: ./images/example_00002.png "Example Traffic Sign"
[image07]: ./images/example_00003.png "Example Traffic Sign"
[image08]: ./images/example_00004.png "Example Traffic Sign"
[image09]: ./images/example_00005.png "Example Traffic Sign"
[image10]: ./images/softmax01.png "Prediction 1"
[image11]: ./images/softmax02.png "Prediction 2"
[image12]: ./images/softmax03.png "Prediction 3"
[image13]: ./images/softmax04.png "Prediction 4"
[image14]: ./images/softmax05.png "Prediction 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the standard python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34,799
* The size of test set is ? 12,630
* The shape of a traffic sign image is ? 32x32
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook.  

For each of 43 classes / labels, I display 10 training data to see samples of image used by model to learn. 

![Visualization of training data][image01]

In the end I show histogram of data samples for each classes. The highest number of samples is class 2: Speed limit (50km/h) with 2010 samples. And the lowest number 
of samples is class 0: Speed limit (20km/h) and class 19: Dangerous curves to the left, with total of 180 samples.

![Training data histogram][image02]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the seventh code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale in order to reduce noise from the images. Detecting traffic signal mainly related to detecting the shape, or edges, of the signal. In most of cases, it doesn't matter about its color variation. Converting to grayscale will also remove other noises such as too dark, too bright, etc. Some of the images used for training is too dark, it will be affecting the model when learning the signal resulting more risk of overfitting. 

Here are examples of a traffic sign image before grayscaling.

![Original Image][image03]

And here are examples of traffic sign image above after grayscalling.

![Grayscaling Image][image04]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.  The code load images from pickled files for each category: Training, Validation, and Test. The validation sets is used to cross validate my model

My final training set had ***34,799*** number of images. My validation set and test set had ***4,410*** and ***12,630*** number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         			|     Description.            						| 
|:------------------------------:|:----------------------------------------------------------------:| 
| Input         			| 32x32x1 RGB image   						| 
| Convolution1: 5x5     	| 1x1 stride, valid padding, outputs 28x28x6		|
| RELU				|										|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6 				|
| Convolution2: 5x5	| 14x14x6, valid padding, outputs  10x10x16	|
| RELU				|										|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16					|
| Fully connected1		| Input = 400. Output = 120       				|
| RELU				| 										|
| Fully connected2		| Input = 120. Output = 84        				|
| RELU				|										|
| Fully connected1		| Input = 84. Output = 43        				|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth cell of the ipython notebook. 

To train the model, I used an EPOCH 20. Each batch size is set to 128. The training rate is set to 0.001. For the optimizer, the Adam optimizer is used. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 0.996
* validation set accuracy of 0.934
* test set accuracy of 0.922

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? It works well with handwritten problem used in MNIST. The traffic sign problem is similar to handwritten problem so I believe LeNet architecture is relevant to traffic sign application. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The accuracy for training, validation, and test are above 0.9
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image05] ![alt text][image06] ![alt text][image07] 
![alt text][image08] ![alt text][image09]



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 44th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        	|     Prediction	        			| 
|:-------------------------------------:|:--------------------------------------------:| 
| No Entry     				| No Entry   					| 
| Roundabout mandatory     	| Roundabout mandator			|
| Keep left				| Keep left					|
| Children crossing	      		| Children crossing				|
| Speed limit (20 km/h)		| Stop		      				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.2%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 47th cell of the Ipython notebook.

For the first image, the model is very sure that this is a ***no entry*** (probability of 0.99), and the image does contain a no entry. The top five soft max probabilities were shown below:

![Prediction 1][image10]

For the second image, the model is very sure that this is a ***roundabout mandatory*** (probability of 0.96), and the image does contain a roundabout mandatory. The top five soft max probabilities were shown below:

![Prediction 2][image11]

For the third image, the model is relatively sure that this is a ***keep left*** (probability of 0.82), and the image does contain a keep left. The top five soft max probabilities were shown below:

![Prediction 3][image12]

For the fourth image, the model is very sure that this is a ***children crossing*** (probability of 0.99), and the image does contain a childresn crossing The top five soft max probabilities were shown below:

![Prediction 4][image13]

For the fifth image, the model is very sure that this is a ***stop sign***  (probability of 0.99), but the image is actually contain a speed limit (20 km/h). The top five soft max probabilities were shown below:

![Prediction 5][image14]

###Analysis to failed prediction to image taken from web

The reason of why prediction for last image that is Speed limite (20km/h) failing because the amount of training set is very less. It is explained in the histogram above, class 0 (speed limit 20km/h) is only 180 samples.  Also the preprocessing of training data still not good enough like no histogram equalization, and no data augmentation for training data that makes recognition to traffic signs containing number is still failing. 