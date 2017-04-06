#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NetworkUsed.PNG "Model Visualization"
[image2]: ./examples/right_2017_04_04_14_32_52_687.jpg "Hard track"
[image3]: ./examples/before.jpg "Recovery Image"
[image4]: ./examples/middle.jpg "Recovery Image"
[image8]: ./examples/end.jpg "Recovery Image"
[image5]: ./examples/cropped.jpg "Cropped image"
[image6]: ./examples/tobeflipped.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The driving works for the easy track as well as for the hard track

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses the Nvidia neural network for self-driving cars. We first normalize the input with a Lambda layer. We then use 3 successive convolutional networks with 5x5 filters with a stride of 2. These are followed by 2 3x3 kernels convilutional networks. We fllatten the output and go through 4 Dense (or Fully Connected) networks with 100, 50 and 10 neurons before entering the output layer.

The model includes RELU layers to introduce nonlinearity (code lines 84, 92, 96, 100). 

####2. Attempts to reduce overfitting in the model

Several Dropouts were introduced (code lines 85, 92, 96, 100) in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also get the right balance of data from the easy and the hard track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a known model, made available to focus mainly on the data and parameters fine-tuning. I thus use the Nvidia CNN architecture they published in "End to End Learning for Self-Driving Cars"

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I ran it through the simulator first to have an idea of the results. It was performing rather poorly on 2 areas of the track where the cars went off road. I looked back at the results of the calculation and found the overfitting was a probable cause of the issue. I then introduced dropouts after each relu and used the classic parameter of 0.5 to keep half of the inputs at each run.

At that point of time, I was using the raw images from the class, only flipping them and using for right and left images a correction of 0.2 for the angle correction.

I tried again and got off track again but the car was driving less erratically. I decided to boost the number of Epochs for the training since after using dropout, the loss was around 0.03. I could safely drive around the "easy" track one with a 30 Epochs trained network.

I then tried this network on the hard track 2 without training it on the the track at all. The car took 2s to go off track. I then tried to up again the number of Epochs to 50 and even 100 but the results were exactly the same.

I then started to collect data from the hard track to be used during a new training. Started with 30 epochs, I got 95% of the hard track covered and got off track on one spot, the easy track was still ok. At the time the loss was a little over 0.02.

I had to boost the number of epochs to 80 to break this 0.02 line and go down to around 0.016. I got around the hard track without any problems then. However, when I tried on the easy track, the car started to act strangely and drove on the right side of the track until it hit a pole there. I tried using a correction for the steering angles of the right and left cameras to 0.5, it improved the situation but I still had off road situations and the car was driving erratically.

There is probably a proper number of epochs that would have worked but I looked at the data instead. The hard track has more features, more turns, less straight lines. Even though I had more data already for the easy track (60/40), I thought the "situations" were much less so I was, with more data, overtraining the hard track. I trained again the easy track to grab more data and ended up with a 75/25 ratio and trained for 50 Epochs.

The car went through both tracks without problems. This can be checked on the video "runOK.mp4"

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture :

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the data provided with the course. I recorded one lap on the hard track after completing succesfully the first track:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image9]



To augment the data sat, I also flipped images so the car would have more sense turning in both direction rather than only on one side:

![alt text][image6]
![alt text][image7]

I also cropped the image in the cropped layer of the network:

![alt text][image5]

After the collection process, I had 50,856 number of data points including right and left cameras. I then preprocessed this data by flipping the center image and got 16,952 more data point. I didn't flip the right and left cameras, it didn't feel useful to do so.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by the fact that with 30 epochs, the car went off road in the track 2 run while if I used 88 or 100 epochs, the car was running track 2 very well but ended up driving on the right side of the road for track 1 or quite erratically, like drunken drive. I would guess that overlearing track 2 the car learned to stay on the right side of the track instead of being in the middle of its track.

Next steps, I would like to use grayscale to see if I can decrease the number of epochs, I also want to try having less data from the easy track where I seem to repeat the same thing (driving in straight line, having a large turn with the same angle).
I used an adam optimizer so that manually training the learning rate wasn't necessary.
