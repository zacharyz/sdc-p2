Writeup:

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/img1result.png "Image 1 Result"
[image5]: ./examples/img2result.png "Image 2 Result"
[image6]: ./examples/img3result.png "Image 3 Result"
[image7]: ./examples/img4result.png "Image 4 Result"
[image8]: ./examples/img5result.png "Image 5 Result"
[image9]: ./examples/trainingsets.png "Traing Sets"
[image10]: ./examples/testimages.png "Test Images"
[image11]: ./examples/testimagesprocessed.png "Test Images Processed"
[image12]: ./examples/initialtraintestchart.png "Distribution of classes in training and test sets"



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



### My trained model correctly identified all 5 of the images I selected.

| Image			    	    |     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Turn Left Ahead     		| Turn Left Ahead   							| 
| No Entry     		    	| No Entry 										|
| Roundabout mandatory	    | Roundabout mandatory							|
| Road narrows on the right	| Road narrows on the right		 				|
| Children Crossing			| Children Crossing   							|

Top 5 Softmax Probabilities for the images:

Image 1 (Turn Left Ahead) 20.7%
![alt text][image4]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .20         			| Turn Left Ahead								| 
| .04     				| Keep Right									|
| .039					| No Passing									|
| -.0077      			| Yield							 				|
| -.0675			    | Ahead only	      							|

Image 2 (No Entry) 21.6%
![alt text][image5]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .21         			| No Entry										| 
| .069     				| Keep left										|
| .022					| Stop											|
| -.01      			| No Passing					 				|
| -.03				    | Ahead only	      							|

Image 3 (Roundabout mandatory) 12.25%
![alt text][image6]

| Probability         	|     Prediction	        						 | 
|:---------------------:|:--------------------------------------------------:| 
| .12         			| Roundabout mandatory								 | 
| .047     				| Right-of-way at the next intersection				 |
| .043					| Speed limit (100km/h)								 |
| .017      			| Priority road					 					 |
| -.007				    | End of no passing by vehicles over 3.5 metric tons |

Image 4 (Road narrows on the right) 36.7%
![alt text][image7]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .367         			| Road narrows on the right						| 
| .335     				| General Caution								|
| .196					| Keep left										|
| .184      			| Pedestrians					 				|
| .122				    | Road work		      							|

Image 5 (Children Crossing) 18.6%
![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .186         			| Children Crossing								| 
| -.046    				| End of all speed and passing limits			|
| -.047					| Bicycles crossing								|
| -.06      			| Right-of-way at the next intersection			|
| -.087				    | No passing	      							|

None of the images scored above 50%. Image 4 - Road Narrows on the Right - for example had two classes within 4% of each other while still being correct. The alternative prediction was General Caution, which under some conditions could look similar. I think this largely comes down to the limited amount of samples I had to work with.  