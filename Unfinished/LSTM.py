print('''\n\n
===================================
==            LSTM Model         ==
===================================''')

print('\nSetting up dependencies...', end='\r')

import os
from pprint import pprint
import numpy as np

from tensorflow.keras.layers import Dense, LSTM, Dropout
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



import Data.process_data_dense as mnist # My data processing module for MNIST.csv

print('Preparing neural network...', end='\r')

model = Sequential([
    LSTM(128, input_shape=(28, 28), activation='relu', return_sequences=True),
    Dropout(0.2),
    
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(10, activation='softmax')
])

print('Compiling neural network...', end='\r')

opt = Adam(lr=1e-3, decay=1e-5)
model.compile(
    optimizer = opt,
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

print('Training neural network...\n')

model.fit(
    np.array(mnist.train_samples).reshape(-1, 28, 28),
    mnist.train_labels,
    validation_split = 0.1,
    epochs = 1
)

print('\n')
print('Testing neural network...', end='\r')



print('Saving neural network... ', end='\r')

model.save('Models/MNIST ConvNet Model.h5')
    
print('Neural network saved!   ')

print('Confusion matrix generated!   ')

print('\nNeural network running complete! ')