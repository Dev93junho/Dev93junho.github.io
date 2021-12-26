# basic Module import
from PIL import Image 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set the dataset directory
train_img_dir = 'D://dataset/handwrite_dataset/train/'
test_img_dir = 'D://dataset/handwrite_dataset/test/'

# load data.csv 
train_csv = pd.read_csv('D:/dataset/handwrite_dataset/train/train_data.csv')
test_csv = pd.read_csv('D:/dataset/handwrite_dataset/test/test_data.csv')

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

# --------------------Start Augmentation --------------------#
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_gen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.6,
    width_shift_range=0.15,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

aug_size = 55000

random_mask = np.random.randint(dataset.shape[0], size=aug_size)
dataset_aug = dataset[random_mask].copy()
dataset_label_aug = dataset_label[random_mask].copy()

dataset_aug = img_gen.flow(dataset_aug, np.zeros(aug_size),
                           batch_size=aug_size, shuffle=False).next()[0]

dataset = np.concatenate((dataset, dataset_aug))
dataset_label = np.concatenate((dataset_label, dataset_label_aug))

print("dataset shape:", dataset.shape)
print("dataset label shape:", dataset_label.shape)

num_sample = 5
random_idxs = np.random.randint(60000, size=num_sample)

plt.figure(figsize=(14,8))
for i, idx in enumerate(random_idxs):
    img = dataset[idx, :]
    label = dataset_label[idx]
    
    plt.subplot(1, len(random_idxs), i+1)
    plt.imshow(img)
    plt.title("Index : {}, Label :{}".format(idx, label))

# ---------------------- End Augmentation ----------------------#


# ------------------------- import gpu -------------------------#
tf.debugging.set_log_device_placement(True)
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
    )
# ------------------------- import gpu -------------------------#

# ## -----------------------------------------------------------#
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

#----------------- set the hyper parameters ------------------#
batch_size = 128
num_classes = 10
epochs = 50 # 정확한 분류를 위해 epoch을 높게 잡아도 됨. 그럼 최적 epoch은?

img_rows = 28 # image size
img_cols = 28
#--------------------------------------------------------------#

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

X = dataset.astype(int)
y = dataset.astype(int)

# import dataset
idx_train, idx_test = train_test_split(range(X.shape[0]), test_size=0.25, random_state=101)

X_train = X[idx_train, :, :, :]
X_test = X[idx_test, :, :, :]
y_train = y[idx_train, :]
y_test = y[idx_test, :]
    
input_shape = (img_rows, img_cols, 1)
x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255. # float64 변경시 precision을 향상 시킬 수 있음
x_test = x_test.astype('float32') / 255. # 다만 float64의 경우 메모리 사용량이 두배로 증가하므로 유의

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# to_categorical 을 사용한다면 다음과 같이 변경해서 사용할 것 ** keras utils => keras.utils import np_utils **
y_train = np_utils.to_categorical(y_train, num_classes) 
y_test = np_utils.to_categorical(y_test, num_classes)

# Build model
model = Sequential()
model.add(Conv2D(784, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (2, 2), activation='softmax', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])


hist = model.fit(
    X_train, y_train, 
    batch_size=batch_size, epochs=epochs, 
    verbose=1, validation_data=(X_test, y_test), 
    shuffle=True)

# print model architecture
model.summary()

# eval model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


import matplotlib.pyplot as plt

predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = np.argmax(y_test, axis=1)

count = 0

plt.figure(figsize=(12,8))

for n in range(16):
    count += 1
    plt.subplot(4, 4, count)
    plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()
