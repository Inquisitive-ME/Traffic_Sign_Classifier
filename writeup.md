# **Traffic Sign Recognition Project** 

## Project Overview

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./PicturesforReadme/PictureExploreTrafSign.png "Picture Visualization"
[image2]: ./PicturesforReadme/HistogramTrafData.png "Histogram Visualization" 

[image3]: ./PicturesforReadme/OrigImagePreProc.png "Image before pre-processing"
[image4]: ./PicturesforReadme/StartPreProcessing.png "First pre-processing attemp"
[image5]: ./PicturesforReadme/FinalPreProcessing.png "Final pre-processing"


[image6]: ./TestImages/5_Speed_Limit_80.jpeg "Traffic Sign 1"
[image7]: ./TestImages/11_Right-of-way_Next_Intercection.jpeg "Traffic Sign 2"
[image8]: ./TestImages/12_Priority_Road.jpeg "Traffic Sign 3"
[image9]: ./TestImages/14_Stop.jpeg "Traffic Sign 4"
[image10]: ./TestImages/25_Road_Work.jpeg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Inquisitive-ME/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* 34799 training examples
* 4410 validation examples
* 12630 testing examples
* The Image data shape is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I did two exploratory visualizations of the dataset. The first is a picture for each unique label in the dataset. This gives a good visual of the quality and type of images in the dataset and can be seen below

![alt text][image1]

The second is a histogram of the dataset labels. This shows that we do not have equal data for each class which could cause issues when training the neural network.
![alt text][image2]


### Design and Test a Model Architecture

#### I started desiging and testing my model architecture with the objective of doing the least amount of preprocessing possible to get acceptable results. My thought was that I could modify the network architecture to make up for issues in the data rather than preprocessing the data, or creating additional data.

I started with the suggested data preprocessing of (pixel - 128)/128. With the LeNet Architecture from the Lenet lab. I started with a learning rate of 0.001, 5 epochs and a batch size of 128. I was able to get an accurcy of 0.907. Below is a picture of one of the signs before and after preprocessing.

** Image Before Pre-Processing **
![alt text][image3]

** Image After Initial Pre-Processing **
![alt text][image4]


My next step was to keep the suggeted preprocessing and see if I could improve the network to get better results

* I started by increasing the width of the convolutional layers. I multiplied the width of the LeNet Architecture by 3 and carried it through all the layers. This increased by accuracy to 0.925

* Next I played around with adding a 1x1 convolution to the begining and adding an extra fully connected layer to the end, but I was not able to get any significant increase in the accuracy and my model training time was significantly increasing so I decided to keep LeNet Architecture and play with the layer sizes

* I found better results by decreasing layer size to the last fully connected layer, and I ended up being able to change the layer sizes of the LeNet network to get an accuracy of 0.951, below is a table with a high level of the different iterations I performed

| Trial Method	      		|    Accuracy Result	        					| 
|:---------------------:|:---------------------------------------------:| 
| Initial Data Preprocession (pixel - 128)/128 with LeNet <br> learning rate = 0.001 <br> epochs = 5 <br> batch size = 128 	| 0.907	|
| triple width of convolutional layers          <br> learning rate = 0.001 <br> epochs = 5 <br> batch size = 128 	| 0.925	|
| triple width of convolutional layers again (x9)<br> learning rate = 0.001 <br> epochs = 5 <br> batch size = 128 	| 0.917	|
| back to only triple width of LeNet with added 1x1 convolution at begining<br> learning rate = 0.001 <br> epochs = 5 <br> batch size = 128 	| 0.934	|
|only triple width of LeNet with added 1x1 convolution at begining and extra fully connected layer<br> learning rate = 0.001 <br> epochs = 5 <br> batch size = 128 	| 0.928	|
| triple width of LeNet with smaller layer size for fully connected layers<br> learning rate = 0.001 <br> epochs = 5 <br> batch size = 128 	| 0.944	|
| triple width of LeNet with smaller layer size for fully connected layers<br> learning rate = 0.0005 <br> epochs = 20 <br> batch size = 64 	| 0.951	|

#### After finding a suitable architecture I wanted to work with preprocessing the images more than just the (pixel-128)/128. It made since to me to scale the images so I decided to use (pixel - min(pixel))/(max(pixel)-min(pixel) which scales each pixel to an intensity of 0 to 1. With this I was able to get to an accuracy of 0.961. below is a picture of one of the signs before and after the final preprocessing

** Image Before Pre-Processing **
![alt text][image3]

** Image After Initial Pre-Processing **
![alt text][image5]


#### 3. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         				|     Description	        						| 
|:---------------------:|:---------------------------------------------:| 
| Input         				| 32x32x3 RGB image   							| 
| Convolution 5x5    	 	| 1x1 stride, same padding, outputs 28x28x18 		|
| RELU					|											|
| Max pooling	      			| 2x2 stride,  outputs 14x14x18 					|
| Convolution 5x5			|  1x1 stride, same padding, outputs 10x10x48     	|
| RELU					|											|
| Max pooling	      			| 2x2 stride,  outputs 5x5x48 					|
| Flatten					| outputs 1200								|
| Fully connected			| linear, outputs 360       						|
| RELU					|											|
| Fully connected			| linear, outputs 100	    						|
| RELU					|											|
| Fully connected			| linear, outputs 43		    						|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used a combination of my personal computer an an amazon web server instance type g2.2xlarge to train the model. The web server was about 5 times faster than my personal computer without a gpu. I recently purchased an nVidia gtx 1080 graphics card which made my computer run the training about 15 times faster.

I started with a learning rate of 0.001, 10 epochs and a batch size of 128. As I got better accuracy I started tuning these values to end up with a learning rate of 0.0005, 40 epochs and a batch size of 64.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of  96.1%
* test set accuracy of 94.6%

I started with the LeNet architecture and minimal preprocessing. I then adjusted the layers of the LeNet architecture because I thought since these images were color I would need to extract 3 times as much information for the 3 color channels.

Once I stopped seeing large changes from changing the architecture, I went back to the preprocessing.I was able to get 

I added in a dropout to try to prevent overfitting.

I think my main issue was that I am overfitting the data. I started thinking that I wanted to do minimum preprocessing, but I ended up with a model that could get 100% of the training set correct while not performing as well on the validation and test sets.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection     		| Right-of-way at the next intersection									| 
| Priority road      			| Priority road  										|
| Speed limit (80km/h)					| Right-of-way at the next intersection											|
| Road work	      		| End of all speed and passing limits					 				|
| Stop			| Speed limit (60km/h)      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is much worse than the accuracy on the test set of 94.6%. This leads me to beleive the model is overfit for the training data. I also think that since I am providing color images the lighting and coloring differences could have a negative effect.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model returned a 100% certainty that the image is the correct Right-of-way at next intercection image. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			|  Right-of-way at the next intersection									| 
| .0     				| Pedestrians 										|
| .0					| Beware of ice/snow											|
| .0	      			| General caution				 				|
| .0				    | End of no passing   							|


For the last image the model was relatively sure that the stop sign was a Speed Limit (60km/h) sign. This is concerning since the model was even more certain on the other incorrect sign predictions.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .845         			| Speed limit (60km/h)   									| 
| .105     				| Speed limit (50km/h)									|
| .04					| Speed limit (70km/h)											|
| .007	      			| No passing				 				|
| .001				    |    Children crossing							|


Overall I think the model was able to get good results form the data by overfitting the data. I think more preprocessing and generating extra data from the signs which had very little data would have greatly improved the results

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

TODO
