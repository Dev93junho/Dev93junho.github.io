from datetime import date
from keras import callbacks
import pandas as pd

# load img
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np

# set the dataset directory
train_img_dir = 'D://dataset/handwrite_dataset/train/'
test_img_dir = 'D://dataset/handwrite_dataset/test/'

# load data.csv 
train_csv = pd.read_csv('D:/dataset/handwrite_dataset/train/train_data.csv')
test_csv = pd.read_csv('D:/dataset/handwrite_dataset/test/test_data.csv')

# # print csv
# print(train)
# print(test)

# save the dataset & label
dataset=[]
dataset_label=[]

# load dataset, save to dataset Array
for file in train_csv['file_name']:
    img_array = np.array(Image.open(train_img_dir+file))
    dataset.append(img_array)

# save the data label
for label in train_csv['label']:
    dataset_label.append(label)
    
# dataset convert to Array from List type
dataset = np.array(dataset)
dataset_label = np.array(dataset_label)

# dataset size convert
dataset = dataset/255.
dataset = dataset.reshape(-1, dataset.shape[1], dataset.shape[2], 1)

# test dataset convert
test_dataset = []

for test_file in test_csv['file_name']:
    image_array = np.array(Image.open(test_img_dir + test_file))
    test_dataset.append(image_array)
    
test_dataset = np.array(test_dataset)

# convert test data size 
test_dataset = test_dataset/255.
test_datset = test_dataset.reshape(-1, test_dataset.shape[1], test_dataset.shape[2], 1)

# split data
from sklearn.model_selection import train_test_split
train_dataset, validation_dataset, train_dataset_label, validation_dataset_label = train_test_split(dataset, dataset_label, test_size = 0.2, stratify = dataset_label)



# Array를 저장할 디렉터리
dataset_dir = "D://dataset/handwrite_dataset/"

# 모든 데이터세트 저장
np.save(dataset_dir + "train_dataset.npy", train_dataset)
np.save(dataset_dir + "train_dataset_label.npy", train_dataset_label)

np.save(dataset_dir + "validation_dataset.npy", validation_dataset)
np.save(dataset_dir + "validation_dataset_label.npy", validation_dataset_label)

np.save(dataset_dir + "test_dataset.npy", test_dataset)

# dataset_loadtest = np.load(dataset_dir + "train_dataset.npy")
# # Array가 잘 불러왔는지 이미지로 확인합니다.
# import matplotlib.pyplot as plt 
# # 첫 번째 이미지를 불러와봅니다.
# plt.imshow(dataset_loadtest[0])

# # import tensorflow
# import tensorflow as tf
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import ModelCheckpoint,EarlyStopping
# from keras.utils import np_utils

# # import gpu
# tf.debugging.set_log_device_placement(True)
# for gpu in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_virtual_device_configuration(
#         gpu,
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
#     )

# ## -------------------------------------------------------------------------------#
# # # sklearn Logistic Regression
# # from sklearn.linear_model import LogisticRegression

# # # calc acc
# # def compute_acc(true,pred):
# #     return sum(true==pred) / len(true)

# # # define model
# # linear_reg = LogisticRegression(solver='liblinear')

# # # sep X, y
# # X = all_images.iloc[:, :-1].astype(int)
# # y = all_images.iloc[:, -1].astype(int)

# # # learn
# # linear_reg.fit(X,y)

# # # predict
# # pred = linear_reg.predict(X)
# # print(compute_acc(y, pred))

# ## -------------------------------------------------------------------------------#
# import os, random
# from glob import glob


# # fixed seed for precision
# def seed_everything(seed=0):
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
    
# seed = 1901
# seed_everything(seed)

# # set the hyper parameters
# batch_size = 128
# num_classes = 10
# epochs = 50 # 정확한 분류를 위해 epoch을 높게 잡아도 됨. 그럼 최적 epoch은?

# img_rows = 28 # image size
# img_cols = 28

# ## -------------------------------------------------------------------------------#

# # import dataset
# (x_train, y_train), (x_test, y_test) = train_images, test_images

# input_shape = (img_rows, img_cols, 1)
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# x_train = x_train.astype('float32') / 255. # float64 변경시 precision을 향상 시킬 수 있음
# x_test = x_test.astype('float32') / 255. # 다만 float64의 경우 메모리 사용량이 두배로 증가하므로 유의

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')


# # to_categorical 을 사용한다면 다음과 같이 변경해서 사용할 것 ** keras utils => keras.utils import np_utils **
# y_train = np_utils.to_categorical(y_train, num_classes) 
# y_test = np_utils.to_categorical(y_test, num_classes)

# # Build model
# model = Sequential()
# model.add(Conv2D(784, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(128, (2, 2), activation='softmax', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # Compile model
# model.compile(
#     loss='categorical_crossentropy', 
#     optimizer='adam', 
#     metrics=['accuracy'])


# hist = model.fit(
#     x_train, y_train, 
#     batch_size=batch_size, epochs=epochs, 
#     verbose=1, validation_data=(x_test, y_test), 
#     callbacks=callbacks ,shuffle=True)

# # print model architecture
# model.summary()

# # eval model
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# import matplotlib.pyplot as plt

# predicted_result = model.predict(x_test)
# predicted_labels = np.argmax(predicted_result, axis=1)

# test_labels = np.argmax(y_test, axis=1)

# count = 0

# plt.figure(figsize=(12,8))

# for n in range(16):
#     count += 1
#     plt.subplot(4, 4, count)
#     plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
#     tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
#     plt.title(tmp)

# plt.tight_layout()
# plt.show()
