import os
import csv

samples = []
with open('.\data\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("Length of train samples ", len(train_samples))
print("Length of validation samples ", len(validation_samples))
import cv2
import numpy as np
import sklearn
from PIL import Image
#import base64
#from io import BytesIO

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size//4]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                image = Image.open(name)
                center_image = np.asarray(image)
                #center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                image_flipped = np.fliplr(image)
                center_image_flipped = np.asarray(image_flipped)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

                name = './data/IMG/' + batch_sample[1].split('/')[-1]
                image = Image.open(name)
                left_image = np.asarray(image)
                left_angle = float(batch_sample[3]) + 0.2
                images.append(left_image)
                angles.append(left_angle)

                name = './data/IMG/' + batch_sample[2].split('/')[-1]
                image = Image.open(name)
                right_image = np.asarray(image)
                right_angle = float(batch_sample[3]) - 0.2
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=8)

ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#input_shape = (160,320,3)
# The network will be a regression not classification so no softmax
# We will remodel the Nvidia network (if we can...)

# Normalization layer
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((60, 20), (0, 0))))

#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
#Convolutional 24@79x159
model.add(Convolution2D(24, 5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#Convolutional 36@39x79
model.add(Convolution2D(36, 5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#Convolutional 48@19x39
model.add(Convolution2D(48, 5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#Convolutional 64@9x19
model.add(Convolution2D(64, 3,3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#Convolutional 64@5x10
model.add(Convolution2D(64, 3,3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples*4), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=30)

from keras.models import load_model

model.save('my_model_test.h5')
