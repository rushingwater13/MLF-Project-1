#import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Read in the data
data = pd.read_csv("Gaussian 2D Wide.csv", header=None)
data = data.values

# Divide the classes
class0 = data[:, 0:2]
class1 = data[:, 2:4]

# Combine the classes into one array
X = np.vstack((class0, class1))

# Create an array to hold the classifications
y_class0 = np.zeros(len(class0))
y_class1 = np.ones(len(class1))

# Add the classifications to the data array
y = np.concatenate((y_class0, y_class1))

# Shuffle to distribute the data
X, y = shuffle(X, y, random_state=42)

print(len(X))
print(len(y))
print(X.shape)
print(y.shape)
