print('Setting up dependencies...', end='\r')

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


from process_data import *

