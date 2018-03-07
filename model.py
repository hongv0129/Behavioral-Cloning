
#==============================================================================
# Import Libraries
#==============================================================================
import os
import csv
import cv2
import numpy as np
import sklearn

from sklearn.utils import shuffle as sklearn_shuffle

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
USE_BATCH_GENERATOR = 0
ANGLE_CORRECTION = 0.24
BATCH_SIZE = 128
EPOCH_COUNT = 15
DROPOUT_RATE = 0.50
LEARNING_RATE = 0.00081
DECAY_RATE = 0.0
VALIDATE_SET_SPLIT_RATIO = 0.15

print("Generator Used?  :", USE_BATCH_GENERATOR)
print("Epoch Count      :", EPOCH_COUNT)
print("Batch Size       :", BATCH_SIZE)
print("Learning Rate    :", LEARNING_RATE)
print("Decay Rate       :", DECAY_RATE)

#==============================================================================
def FetchRecordList():
    Lines = []
    with open('./simulation_data/driving_log.csv') as csvFile:
        Reader = csv.reader(csvFile)
        for Line in Reader:
            Lines.append(Line)
    return Lines

def LoadEntireDataset(Lines):
    #center,left,right,steering,throttle,brake,speed
    Images = []
    SteerAngles = []
    for Line in Lines:
        for CameraIndex in range(0, 3, 1):
            SourcePath = Line[CameraIndex]
            FileName = SourcePath.split("\\")[-1]
            RelativePath = './simulation_data/IMG/' + FileName
            #print(RelativePath)
            
            OrgImage = cv2.imread(RelativePath)
            Image = cv2.cvtColor(OrgImage, cv2.COLOR_BGR2RGB)
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
    
    return X_Train, y_Train
#==============================================================================


#==============================================================================
# Model Regression Neural Network
#==============================================================================

def ResizeImage_lambda(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (48, 160))
"""
Model = Sequential()
Model.add(Cropping2D(cropping=((44,20), (0,0)), input_shape=(160,320,3)))
Model.add(Lambda(ResizeImage_lambda))
"""

def CreateNvidiaModel():
    Model = Sequential()
    Model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    Model.add(Lambda(lambda x: x/255.0 - 0.5))
    
    Model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid'))
    Model.add(Activation('relu'))
        #Model.add(MaxPooling2D((2, 2), border_mode='valid'))
        #Model.add(Dropout(DROPOUT_RATE))

    Model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid'))
    Model.add(Activation('relu'))
        #Model.add(MaxPooling2D((2, 2), border_mode='valid'))
        #Model.add(Dropout(DROPOUT_RATE))

    Model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid'))
    Model.add(Activation('relu'))
        #Model.add(MaxPooling2D((2, 2), border_mode='valid'))
        #Model.add(Dropout(DROPOUT_RATE))

    Model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid'))
    Model.add(Activation('relu'))
        #Model.add(Dropout(DROPOUT_RATE))

    Model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid'))
    Model.add(Activation('relu'))
        #Model.add(Dropout(DROPOUT_RATE))

    Model.add(Flatten())

    Model.add(Dense(100))
    Model.add(Activation('relu'))
    Model.add(Dropout(DROPOUT_RATE))

    Model.add(Dense(50))
    Model.add(Activation('relu'))
    Model.add(Dropout(DROPOUT_RATE))

    Model.add(Dense(10))
    Model.add(Activation('relu'))
    Model.add(Dropout(DROPOUT_RATE))

    Model.add(Dense(1))

    Model.summary()
    
    return Model;

#==============================================================================
# Compile & Train Regression Neural Network
#==============================================================================

if os.path.exists("model.h5") == True :
    print("Loading existent Model...")
    Model = load_model("model.h5")
else:
    Model = CreateNvidiaModel()
    print("Generating Model...")
    
AdamOptimizer = KerasOptimizer.Adam(lr=LEARNING_RATE, decay=DECAY_RATE)
Model.compile(optimizer=AdamOptimizer, loss='mse')

EarlyStopper = EarlyStopping(monitor='val_loss', min_delta=0.0006, patience=2, verbose=0, mode='auto')

if(USE_BATCH_GENERATOR == 0):
    ListRecords = FetchRecordList()
    X_Train_tmp, y_Train_tmp = LoadEntireDataset(ListRecords)
    X_Train, y_Train = sklearn_shuffle(X_Train_tmp, y_Train_tmp)
    del X_Train_tmp
    del y_Train_tmp
    HistoryObject = Model.fit(  X_Train, y_Train, 
                                batch_size = BATCH_SIZE, nb_epoch=EPOCH_COUNT,
                                validation_split=VALIDATE_SET_SPLIT_RATIO, shuffle=True, 
                                callbacks=[EarlyStopper] )

Model.save('model.h5')
del Model  # deletes the existing model

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

