
#==============================================================================
# Import Libraries
#==============================================================================
import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, Flatten, Dropout

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import matplotlib.pyplot as plt

import keras.optimizers as KerasOptimizer

## Tool Environment Version Check ##
"""
print('========1==================')
import keras
print(keras.__version__)
print('========2==================')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('========3==================')
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print('========4==================')
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
print('===========================')
"""

#==============================================================================
ANGLE_CORRECTION = 0.24
BATCH_SIZE = 128
EPOCH_COUNT = 13
DROPOUT_KEEP_RATE = 0.4
LEARNING_RATE = 0.00088
DECAY_RATE = 0.0
VALIDATE_SET_SPLIT_RATIO = 0.15

print("Epoch Count      :", EPOCH_COUNT)
print("Batch Size       :", BATCH_SIZE)
print("Learning Rate    :", LEARNING_RATE)
print("Decay Rate       :", DECAY_RATE)
#==============================================================================
Lines = []

#center,left,right,steering,throttle,brake,speed

with open('./simulation_data/driving_log.csv') as csvFile:
    Reader = csv.reader(csvFile)
    for Line in Reader:
        Lines.append(Line)
        
Images = []
SteerAngles = []
for Line in Lines:
    for CameraIndex in range(0, 3, 1):
        SourcePath = Line[CameraIndex]
        FileName = SourcePath.split("/")[-1]
        RelativePath = './simulation_data/IMG/' + FileName
        #print(RelativePath)
        
        Image = cv2.imread(RelativePath)
        Images.append(Image)
        
        SteerAngle = float(Line[3])
        if(CameraIndex == 1): #Left Camera
            SteerAngle = SteerAngle + ANGLE_CORRECTION
        elif(CameraIndex == 2): #Right Camera
            SteerAngle = SteerAngle - ANGLE_CORRECTION
        SteerAngles.append(SteerAngle)
        
        MirrorImage = cv2.flip(Image, 1)
        MirrorSteerAngle = -SteerAngle
        
        Images.append(MirrorImage)   
        SteerAngles.append(MirrorSteerAngle)

print("Number of Training Data: ", len(Images))
assert(len(Images) == len(SteerAngles))

X_Train = np.array(Images)
y_Train = np.array(SteerAngles)
print(X_Train.shape)

#==============================================================================
# Model Regression Neural Network
#==============================================================================

"""
def ResizeImage_lambda(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (48, 160))
    
Model = Sequential()
Model.add(Cropping2D(cropping=((44,20), (0,0)), input_shape=(160,320,3)))
Model.add(Lambda(ResizeImage_lambda))
"""

Model = Sequential()
Model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

Model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
Model.add(Activation('relu'))
    #Model.add(MaxPooling2D((2, 2), border_mode='valid'))
    #Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
Model.add(Activation('relu'))
    #Model.add(MaxPooling2D((2, 2), border_mode='valid'))
    #Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
Model.add(Activation('relu'))
    #Model.add(MaxPooling2D((2, 2), border_mode='valid'))
    #Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid'))
Model.add(Activation('relu'))
    #Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid'))
Model.add(Activation('relu'))
    #Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Flatten())

Model.add(Dense(100))
Model.add(Activation('relu'))
Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Dense(50))
Model.add(Activation('relu'))
Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Dense(10))
Model.add(Activation('relu'))
Model.add(Dropout(DROPOUT_KEEP_RATE))

Model.add(Dense(1))

Model.summary()

#==============================================================================
# Compile & Train Regression Neural Network
#==============================================================================
AdamOptimizer = KerasOptimizer.Adam(lr=LEARNING_RATE, decay=DECAY_RATE)
Model.compile(optimizer=AdamOptimizer, loss='mse')

HistoryObject = Model.fit(X_Train, y_Train, batch_size = BATCH_SIZE, 
                            validation_split=VALIDATE_SET_SPLIT_RATIO, 
                            shuffle=True, nb_epoch=EPOCH_COUNT)
Model.save('model.h5')

#==============================================================================
# Visualize Loss Metrics
#==============================================================================

# print the keys contained in the history object
print(HistoryObject.history.keys()) 

# plot the training and validation loss for each epoch
plt.plot(HistoryObject.history['loss']) 
plt.plot(HistoryObject.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
#------------------------------------------------------------------------------

