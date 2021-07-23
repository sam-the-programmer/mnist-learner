print('''\n\n===========================
==      Dense Model      ==
===========================''')

print('\nSetting up dependencies...', end='\r')

import os
import random
import numpy as np

# Prevent warnings if you haven't got NVIDIA CUDA toolkit
# (I have not configured this for GPU running)

os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '3' 
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '0' # Reset console display settings

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


print('Dependencies imported!     ')

print('Generating methods and commands... ', end='\r')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without Normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show_data(array):
    plt.imshow(array.reshape(28, 28), cmap=plt.cm.Blues)
    plt.show()

print('Methods and commands generated!   ')



import Data.process_data_dense as mnist # My data processing module for MNIST.csv

print('Preparing neural network...', end='\r')

model = Sequential([
    Dense(units = 32, activation = 'relu', input_shape = (784,)),
    Dense(units = 16, activation = 'softplus'                  ),
    Dense(units = 16, activation = 'relu'                      ),
    Dense(units = 10, activation = 'softmax'                   )
])

print('Compiling neural network...', end='\r')

model.compile(
    optimizer = 'Adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# SOME EDITED STUFF HERE

print('Training neural network...\n')
history = model.fit(
    mnist.train_samples,
    mnist.train_labels,
    batch_size = 64,
    validation_split = 0.1,
    shuffle = True,
    epochs = 50,
    verbose = 0
)
print('\n')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.grid()
# plt.show()

print('Testing neural network...', end='\r')

piece = random.randint(0, 100)
predictions = model.predict(mnist.train_samples,verbose=1)
show_data(mnist.train_samples[piece])
#print(str(mnist.train_samples[piece]).replace('0', ' '))
print(f'\nIt is a {np.argmax(predictions[piece])}\n')


print('Saving neural network... ', end='\r')

model.save('Models/MNIST Dense Model.h5')
    
print('Neural network saved!   ')


print('Confusion matrix generated!   ')

print('\nNeural network running complete! ')