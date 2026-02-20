import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras import models

# Read in the data, don't forget to capture the first row
data = pd.read_csv("Moons 2D Overlap.csv", header=None)
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



# Divide into train + validate and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Now divide into train and validate
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)



# Set up the model, using tensorflow keras
# Sequential because Feed Forward
# ReLU in the hidden layer because the data set is needs curvature
# Sigmoid for the output layer because this is binary classification
model = models.Sequential([
    layers.Dense(units=16, activation='relu', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])


# Compile the model
# Adam optimizer for gradient descent, seems liek the default
# Binary Crossentropy because binary classification

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Train the model
model.fit(X_train, 
    y_train, 
    epochs=200, 
    validation_data=(X_val, y_val)
)


# Evaluate the model on the test data
results = model.evaluate(X_test, y_test)

print(results)

# what kind of data do we need to report?
# so confusion matrix



# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.freecodecamp.org/news/binary-classification-made-simple-with-tensorflow/
# https://keras.io/guides/training_with_built_in_methods/