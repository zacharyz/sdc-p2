# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/img1result.png "Image 1 Result"
[image5]: ./examples/img2result.png "Image 2 Result"
[image6]: ./examples/img3result.png "Image 3 Result"
[image7]: ./examples/img4result.png "Image 4 Result"
[image8]: ./examples/img5result.png "Image 5 Result"
[image13]: ./examples/img6result.png "Image 6 Result"
[image14]: ./examples/img7result.png "Image 7 Result"
[image15]: ./examples/img8result.png "Image 8 Result"
[image16]: ./examples/img9result.png "Image 9 Result"
[image17]: ./examples/img10result.png "Image 10 Result"
[image9]: ./examples/trainingsets.png "Traing Sets"
[image10]: ./examples/testimages.png "Test Images"
[image11]: ./examples/testimagesprocessed.png "Test Images Processed"
[image12]: ./examples/initialtraintestchart.png "Distribution of classes in training and test sets"
[image18]: ./examples/testimages2a.png "Test Images"
[image19]: ./examples/testimages2b.png "Test Images"
[image20]: ./examples/testimages2c.png "Test Images"




### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

Number of training examples = 34799

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43

Augmented Data:

Number of training examples = 50879

I then used train_test_split with a 80/20 split resulting in:

Size of training set = (40703, 32, 32, 1)

Size of validation set = (10176, 32, 32, 1)

#### 2. Include an exploratory visualization of the dataset.

![alt text][image12]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

I converted to grayscale and scaled of pixel values to [0, 1] (by default they were in [0, 255] range), and represent labels in a one-hot encoding and shuffled.

#### Before Processing:
![alt text][image10]

#### After Processing:
![alt text][image11]

I experimented quite a bit with the use of dropout to prevent overfitting though prior to augmenting my data I actually saw better results without dropout. 

I initially found my final testing result accuracy to be lower than I desired so I then experimented with augmenting my data. This involved duplicating images with small variances in position and rotation. This extended my testing set by ~15k, which allowed me to take advantage of dropout. Additionally the initial data did not have an equal representation of classes, making its effectiveness at training for some classes of signs difficult.

![alt text][image9]

One step I could take to further augment my data would be take classes of signs that could possibly represent other classes and flip or rotate them. For example a sign for Turn Right Head could easily be flipped and duplicated then represent a sample for Turn left ahead.

I could also further normalize my samples by accounting for positioning and rotation of the initial test samples.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

I built upon the LeNet CNN with the addition of dropout and l2 regularization.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 	    	| 1x1 stride, same padding, outputs 28x28x6 	|
| Activation - RELU     |												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 		    | outputs 10x10x16      						|
| Activation - RELU     |												|
| Max pooling   		| outputs 5x5x16								|
| Flatten				| outputs 400									|
| Fully Connected		| outputs 120									|
| Activation - RELU     |												|
| Dropout				| with 0.5 probability							|
| Fully Connected		| outputs 84									|
| Activation - RELU     |												|
| Dropout				| with 0.5 probability							|
| Fully Connected		| outputs 43									|


### Regularization:

#### Dropout
* performed on fully connected layers
* keep_prob of 0.5.
* prevent overfitting and increase generalization
* During training only keep a neuron activate 50% of the time.

#### L2 Regularization
* lamba of 0.00001
* Applied only to weights
* Discouraged large weights through an elementwise quadratic penalty, encouraging the network to use all of its inputs a little rather that some of its inputs a lot.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

### My Final Hyper Parameters
* learning rate             =  0.001
* batch size                =  128
* total epochs              =  100
* dropout keep probability  =  0.5
* l2 lambda  				=  0.00001
* Grayscale 			 	=  True


I initially began with an epoch count of 10, batch size of 128 and learning rate of 0.001. I am training a fairly beefy system with an nvidia 1080 ti so it wasn't much an issue extending the training out to 100+ for experimentation purposes and lower learning rates.

When I thought I had a good combination I would enable model restoration and continue training until I reached validation accuracy of ~98%. Despite the high validation accuracy though my final testing accuracy was below the desired 93%. 

As I described in my data augmentation section above, it wasn't until this point that I decided I was probably overfitting my training and needed more samples. I also settled on basic normalization and grayscaling of my sample images. I initially designed my model to be configurable with the number of input channel (rgb vs grayscale) and the use of dropout and l2 reg. Once I increased my learning rate back to the initial 0.001 with dropout and l2 regularization my training quickly converged to 98%+ within 20 epochs of testing.

For my optimizer I stuck with the cs231n's recommended adam optimizer (http://cs231n.github.io/neural-networks-3/) for adaptive learning rate as well as their recommendation of 0.5 for dropout probability.

### My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 99.3%
* test set accuracy of 94.2%


### Test a Model on New Images

I chose 10 images from around the web. Here are the 10 images.
![alt text][image18]
![alt text][image19]
![alt text][image20]

I chose these images because they represented a variety of lighting levels as well as visual clarity.
The `Speed limit (60km/h) ` and `General Caution` images for example are exceptionally dark and have low contrast. The `Priority road` and `No passing` are both hard to read with my own eyes.

### My trained model correctly identified all 10 of the images I selected.

| Image			    	    				|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)  					| Speed limit (60km/h)   						| 
| No passing  		    					| No Passing									|
| Priority road	    						| Priority road									|
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited		|
| General Caution							| Children Crossing   							|
| Turn Left Ahead  					   		| Turn Left Ahead   							| 
| No Entry     		   					 	| No Entry 										|
| Roundabout mandatory	 				    | Roundabout mandatory							|
| Road narrows on the right					| Road narrows on the right		 				|
| Children Crossing							| Children Crossing   							|



![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

Given the degree of accuracy that my model was able to predict the new images `100%` versus my with my testing set which was only `94.2%` indicates that my model was well trained. If the level of prediction was much lower than my testing accuracy it would have suggested that I had overfitted to the training data. This suggests that my use of data augmentation was effective in expanding my training data.

### Top 5 Softmax Probabilities for the images:
```
INFO:tensorflow:Restoring parameters from .\traffic_sign
TopKV2(values=array([[  9.99999523e-01,   4.45272718e-07,   3.33199406e-11,
          5.40262496e-12,   2.15472497e-12],
       [  1.00000000e+00,   1.30407906e-19,   1.11445113e-19,
          8.78004661e-22,   1.26631889e-23],
       [  1.00000000e+00,   1.03696074e-09,   6.11643791e-11,
          1.25060474e-11,   1.20529897e-11],
       [  1.00000000e+00,   3.96371103e-10,   1.50260540e-10,
          1.13489072e-13,   5.86921835e-16],
       [  9.95429158e-01,   4.53382405e-03,   3.00612901e-05,
          4.28498106e-06,   1.28526517e-06],
       [  1.00000000e+00,   3.48957463e-09,   4.64835948e-11,
          1.33483600e-12,   1.11309494e-13],
       [  9.99927044e-01,   7.29266249e-05,   8.08302817e-12,
          1.24708772e-12,   5.71945811e-13],
       [  9.99831438e-01,   7.29601306e-05,   4.55668051e-05,
          4.30406508e-05,   6.18614104e-06],
       [  9.99993682e-01,   6.10420147e-06,   2.59968800e-07,
          1.98589167e-09,   4.21203454e-11],
       [  9.85637128e-01,   1.30415028e-02,   7.88158504e-04,
          4.04213672e-04,   6.56075063e-05]], dtype=float32), indices=array([[ 3,  5,  2, 23, 13],
       [ 9,  3, 41, 35, 10],
       [12,  9, 41, 17, 40],
       [16,  7,  9, 10,  5],
       [18, 26, 40, 20, 24],
       [34, 35, 30, 13, 38],
       [17, 14, 38, 34, 12],
       [40,  7, 12, 16, 11],
       [24, 26, 18, 28, 27],
       [28, 18, 26, 27, 11]]))
```

The code for generating this is found in the last cell of the notebook. It shows that for each image it was able to effectively predict the probability of the correct class of each image within 0.98-0.99.