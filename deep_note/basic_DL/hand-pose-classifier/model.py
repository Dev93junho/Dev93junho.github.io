import numpy as np
import pandas as pd
from glob import glob
import cv2
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
import pickle
import sys, os, random
import pathlib


# load dataset from D-drive
train_files = sorted(glob('D://dataset/handwrite_dataset/train/*'))
test_files = sorted(glob('D://dataset/handwrite_dataset/test/*'))

# # load dataset from tensorflow dataset 
# (x_train, y_train), (x_test, y_test)=keras.dataset.add()

# import gpu
tf.debugging.set_log_device_placement(True)
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    )


# import data
img_rows = 28
img_cols = 28

(x_train, y_train) = train_files
(x_test, y_test) = test_files

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float64') / 255.
x_test = x_test.astype('float64') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train smaples')
print(x_test.shape[0], 'test samples')

# learning rate
batch_size= 128
num_classes = 10
epochs = 5

# y_train
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# build model
model = Sequential()
model.add(Conv2D(input_shape=(img_rows, img_cols), padding='same', activation='relu'))
model.add(MaxPooling2D)
model.add(Conv2D(activation='relu'))
model.add(MaxPooling2D)
model.add(Conv2D(activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten)
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test) )

# predict result




# accuracy

def ACCURACY(true, pred):   
    score = np.mean(true==pred)
    return score


print('score')