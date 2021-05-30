print('\nSetting up dependencies...', end='\r')

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '0'

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

print('Dependencies imported!     ')

import Data.process_data as mnist

print('Preparing neural network...', end='\r')

model = Sequential([
    Dense(units = 16, activation = 'relu', input_shape = (784,)),
    Dense(units = 16, activation = 'relu'                      ),
    Dense(units = 10, activation = 'softmax'                   )
])

print('Compiling neural network...', end='\r')

model.compile(
    optimizer = 'Adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

print('Training neural network...')
model.fit(
    mnist.train_samples,
    mnist.train_labels,
    batch_size = 64,
    validation_split = 0.1,
    shuffle = True,
    epochs = 50,
    verbose = 2
)

print('Testing neural network...', end='\r')



print('Saving neural network... ')
print('Neural network saved!   ')

print('Generating confusion matrix...')
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
    
    
# cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

# cm_plot_labels = ['No Side Effects', 'Had Side Effects']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels)
print('Confusion matrix generated!   ')

print('\nNeural network complete! ')