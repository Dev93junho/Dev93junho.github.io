import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
import functools
import os

# load dataset in D drive
TRAIN_DATA_URL = "D://dataset/handwrite_dataset/train/train.csv"
TEST_DATA_URL = "D://dataset/handwrite_dataset/test/test.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("test.csv", TEST_DATA_URL)

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

(x_train, y_train) = TRAIN_DATA_URL
(x_test, y_test) = TEST_DATA_URL

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
model.add(Conv2D)
model.add(MaxPooling2D)
model.add(Dropout(0.5))
model.add(Flatten)
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='relu'))
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test) )

# predict result




# accuracy
import numpy as np

def ACCURACY(true, pred):   
    score = np.mean(true==pred)
    return score


print('score')