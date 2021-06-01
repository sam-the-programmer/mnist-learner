print('''\n\n===========================
== Keras SimpleRNN Model ==
===========================''')

print('\nSetting up dependencies...', end='\r')

import os
from pprint import pprint
import numpy as np

# Prevent warnings if you haven't got NVIDIA CUDA toolkit
# (I have not configured this for GPU running)

os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '3' 
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '0' # Reset console display settings

from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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



import Data.process_data_rnn as mnist # My data processing module for MNIST.csv

print('Preparing neural network...', end='\r')

model = Sequential([
    LSTM(units=16, input_shape=(392, 2)),
    Dense(units = 10, activation = 'softmax'                   )
])

print('Compiling neural network...', end='\r')

model.compile(
    optimizer = 'Adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

print('Training neural network...\n')
model.fit(
    mnist.train_samples,
    mnist.train_labels,
    batch_size = 64,
    validation_split = 0.1,
    shuffle = True,
    epochs = 50,
    verbose = 2
)
print('\n')
print('Testing neural network...', end='\r')



print('Saving neural network... ', end='\r')

# if os.path.isfile('Models/MNIST Dense Model.h5') is False: # Only if you want to only save once
model.save('Models/MNIST Dense Model.h5')
    
print('Neural network saved!   ')


print('Generating confusion matrix...', end='\r') # Not implemented yet

show_data(mnist.train_samples[0]) # Just to test whether this works



# cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

# cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels)


print('Confusion matrix generated!   ')

print('\nNeural network running complete! ')