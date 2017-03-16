# **Traffic Sign Recognition** 

## Project Report

---

The goals  of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/sign-distribution.png "Distribution"
[image2]: ./images/random-signs.png "Random Signs"
[image3]: ./images/augmented-signs.png "Augmented Signs"
[image4]: ./images/processed-sign.png "Processed Sign"
[image5]: ./images/test-images.png "Test Images"
[image6]: ./images/softmax-children.png "Softmax Children Crossing"
[image7]: ./images/softmax-noentry.png "Softmax No Entry"
[image8]: ./images/softmax-nopassing.png "Softmax No Passing"
[image9]: ./images/softmax-pedestrian.png "Softmax Pedestrian"
[image10]: ./images/softmax-speedlimit.png "Softmax Speed Limit 30 km/h"

---


## Data Set Summary & Exploration

Here is the [link](https://github.com/leoguim/Traffic_Sign_Classification/blob/master/Traffic_Sign_Classifier.ipynb) to the project code.

### 1. Basic summary of the data set 

The code for this step is contained in the second code cell of the IPython notebook.  

The numpy library was used to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34,799
* Number of validation examples = 4,410
* Number of testing examples = 12,630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 2. Exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third to sixth code cells of the IPython notebook, and it's brokedown in:
1. Functions to add the label names to the labels vector and plot random images with their labels

2. Pandas dataframe and counts the frequency of each labels in the training set.

#### Exploratory visualizations:

![alt text][image1]

   ###### * Bar chart with frequency counts of each label at the training data set

![alt text][image2]   

   ###### * Random plot of signs and their labels


## Design and Test a Model Architecture

### 1. Image preprocessing

The code for this step is contained in the seventh to thirteen code cells of the IPython notebook, and it's brokedown into the following steps:

1. The number of images from the training set was augmented by applying transformations techniques such as random rotation, translation and brightness. Each image from the original training set was augmented by a factor of 3.

![alt text][image3]

   ##### Random plot of augmented images

2. After number augmentatation, all images were converted to grayscale and normalized. The new statistics are as follow:
    * Number of training examples = 139,196
    * Number of validation examples = 4,410
    * Number of testing examples = 12,630
    * Image data shape = (32, 32, 1)
    * Number of classes = 43

![alt text][image4]

   ##### * Random plot of images before and after processing

### 2. Model Architecture

A Lenet Model was used with the following layers. The code can be found at the fifteenth cell of the IPython notebook

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image   							    | 
| Convolution        	| 5x5 kernel, 1x1 stride, valid padding, output = 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 kernel, 2x2 stride, valid padding, output = 14x14x32 				|
| Convolution        	| 5x5 kernel, 1x1 stride, valid padding, output = 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 kernel, 2x2 stride, valid padding, output = 5x5x64 				|
| Fully connected		| output = 120    							    |
| RELU	                |                                               |
| Dropout     		    | Keep Probability = 50%					    |
| Fully connected		| output = 400    							    |
| RELU				    |        									    |
| Dropout				| Keep Probability = 50%						|
| Fully connected		| output = 43    							    |
| SOFTMAX				|                                             |  |



### 3. Training the Model

The code for training the model is located in the nineteenth cell of the ipython notebook. 

The following paramaters were used to train the model:

* **Optimizer** = Adam Optimizer
* **Learning Rate** = 0.001
* **Epochs** = 50
* **Batch Size** = 512
* **DropOut** = 50%

### 4. Approach to solution

The code for calculating the accuracy of the model is located in the eighteenth cell of the Ipython notebook.

The final model results were:
* training set accuracy of **99.9%**
* validation set accuracy of **98.1%**
* test set accuracy of **96.8%**

The following indicates the steps taken during the development of the Model:
1) A Lenet architecture was selected using the parameters as mentioned above, except for the first 2 convolutional layers where an output of 6 and 16 respectively were used. COV=>RELU=>POOL=>COV=>RELU=>POOL=>FC=>RELU=>FC=>RELU=>FC=>SOFTMAX.
2) An initial testing accuracy of 92.4% was measured, without adding any augmented images.
3) The same model was run again using augmented images. In this scenario, the testing accuracy and validation accuracy were less than the initial model, due to overfitting. 
4) The next step was to add Dropouts on the last 2 fully connected layers, and the testing accuracy was increased to 94.8%.
5) The last step was to to increase the output layers of the convolutional layers from 6 to 32 and 16 to 64. After this adjustments, an accuracy of 96.8% was measured on the testing set.
 

## Test a Model on New Images

### 1. Select random images found on the web

Below are five German traffic signs found on the web and used for testing. On previous tests using different model architectures and hyperparameters, the following image misclassifications were observed:
1) Speed limits - speed limit signs were commonly misclassified.  The 30 km/h image was misclassified with 50 km/h or 80 km/h.
2) Children crossing & pedestrians - previous architectures failed to generalized the traffic shape and the content of these 2 signs, and misclassified them with similar signs. 

![alt text][image5]


### 2. Model's prediction description

The code for making predictions is located in the 25th and 26th cells of the Ipython notebook.


| Image			        |     Prediction	        					|    Status     |
|:---------------------:|:---------------------------------------------:| :------------:|
| Pedestrian  	        | Pedestrian  									| OK            |
| No Passing    		| No Passing 								    | OK            |
| Children Crossing		| Children Crossing						| OK            |
| No Entry      		| No Entry					 				    | OK            |
| 30 km/h		        | 30 km/h    							        | OK            |


The model was able to correctly predict 5 of the 5 traffic signs, which gives an accuracy of 100%, compared to the test set accuracy of 96.8%. As seen below on the Top Softmax probabilities, the model assign a near 100% probability to each image.  *Note that since only 5 images were used for the test, this cannot be a generalized statement, misclassifying only one image would have taken the accuracy to 80%. 

### 3. Top softmax probabilities for each text images

The code for making predictions on the final model is located in the 27th and 28th cells of the Ipython notebook.

Below are top 5 softmax probabilities for the testing signs.  From the plots below, it can be seen that the model is very confident (close to 100%) about its prediction for the 5 images.  

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
