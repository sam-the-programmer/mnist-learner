print('\nSetting up dependencies...', end='\r')

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '3'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']  =  '0'

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
print('Dependencies imported!')

from Data.process_data import *

print('Preparing neural network...', end='\r')
model = Sequential([
    Dense(units = 16, activation = 'relu', input_shape = (784,)),
    Dense(units = 16, activation = 'relu'                      ),
    Dense(units = 10, activation = 'relu'                      )
])

print('Compiling neural network...', end='\r')

model.compile(
    optimizer = 'Adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

print('Training neural network...', end='\r')

print('Testing neural network...', end='\r')

print('Neural network complete! ')