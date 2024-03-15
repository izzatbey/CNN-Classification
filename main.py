# .\CNN-venv\Scripts\activate
# deactivate
# pip freeze > requirements.txt
#
# pip install -r requirements.txt

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential

import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras import utils

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# from sklearn import metricsfrom 
from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_splitimport 

model = Sequential()

data_dir = 'Dataset'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

img = cv2.imread(os.path.join(data_dir, 'Normal', '2.jpg'))
img.shape

plt.imshow(cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE))
plt.imshow(img, cmap='gray_r')
plt.show()

# Build dataset with classes on the fly with Keras
train_data = tf.keras.utils.image_dataset_from_directory(data_dir)

train_data_iterator = train_data.as_numpy_iterator()
train_batch = train_data_iterator.next()

test_data = tf.keras.utils.image_dataset_from_directory(data_dir)
test_data_iterator = test_data.as_numpy_iterator()
test_batch = test_data_iterator.next()

train_batch[0].shape
train_batch[1]

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(train_batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(train_batch[1][idx])
    
# Applying scaling to X as X is images and y is labels.
train_data = train_data.map(lambda x, y: (x/255, y))
test_data = test_data.map(lambda x, y: (x/255, y))

train_data.as_numpy_iterator().next()[0].max()
print("Training dataset has %s batches" %(len(train_data)))
print("Testing dataset has %s batches" %(len(test_data)))


# Build dataset with classes on the fly with Keras
train_data = tf.keras.utils.image_dataset_from_directory(data_dir)
train_data_iterator = train_data.as_numpy_iterator()
train_batch = train_data_iterator.next()
test_data = tf.keras.utils.image_dataset_from_directory(data_dir)
test_data_iterator = test_data.as_numpy_iterator()
test_batch = test_data_iterator.next()

train_batch[0].shape
train_batch[1]
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(train_batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(train_batch[1][idx])
    
# Applying scaling to X as X is images and y is labels.
train_data = train_data.map(lambda x, y: (x/255, y))
test_data = test_data.map(lambda x, y: (x/255, y))

train_data.as_numpy_iterator().next()[0].max()
print("Training dataset has %s batches" %(len(train_data)))
print("Testing dataset has %s batches" %(len(test_data)))

# Build validation dataset from the training dataset by allocating 2 batches for the validation partition
train_size = len(train_data)
validation_size = 2

val_data = train_data.take(validation_size)
train_data = train_data.skip(validation_size)

# Input layer with 16 filters of 3x3 pixels size and 1 pixel of stride
model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train_data, epochs=3, validation_data=val_data, callbacks= [tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='loss')
plt.plot(hist.history['val_loss'], color='red', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()



