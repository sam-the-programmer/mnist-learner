print('Setting up dependencies for data processing...', end='\r')
import csv
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

raw = []
train_labels = []
# test_samples = []
# test_labels = []
scaled_train_samples = []


print('Collecting data...                            ', end='\r')
with open('Data/MNIST.csv') as file:
    csv_data = csv.reader(file, delimiter=',')

    raw.extend(iter(csv_data))
print('Parsing data...   ', end='\r')
raw.pop(0)  # Remove headers from the dataset

for image in raw:                     # Iterate through each image
    train_labels.append(image[0])     # Add label to label list
    image.pop(0)                      # Remove from list of the image

train_samples = list(raw)
for train_sample in train_samples:
    for pixel in range(len(train_sample)):
        train_sample[pixel] = int(train_sample[pixel])

# Turn all data and labels into integers
for label in range(len(train_labels)):
    
    if type(train_labels[label]) == list:

        for i in range(len(train_labels[label])):
            train_labels[label][i] = int(train_labels[label][i])

    elif type(train_labels[label]) in [int, str]:
        train_labels[label] = int(train_labels[label])

    else: raise TypeError(f'Label data type {str(type(train_labels[label]))[8 : len(str(type(train_labels[label])))-2]} is not allowed.')


print('Shaping data...', end='\r')

del train_labels[len(train_labels) - 335:]    # Get right size
del train_samples[len(train_samples) - 335:]  # Get right size


assert len(train_samples) == 19600
assert len(train_labels) == 19600

# assert len(test_samples) == 19600
# assert len(test_labels) == 19600

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 784))


print('Data preprocessed!')
if __name__ == '__main__': print(train_samples)