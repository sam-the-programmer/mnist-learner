print('Setting up dependencies for data processing...', end='\r')
import csv
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

raw = []
train_samples = []
train_labels = []
scaled_train_samples = []

print('Collecting data...', end='\r')

print('Parsing data...', end='\r')

print('Shaping data...', end='\r')

print('Data preprocessed!')