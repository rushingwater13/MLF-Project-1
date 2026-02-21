import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras import models

# # Read in the data, don't forget to capture the first row
# data = pd.read_csv("Gaussian 2D Wide.csv", header=None)
# #data = pd.read_csv("Moons 2D Overlap.csv", header=None)
# data = data.values

g_2_w = pd.read_csv("Gaussian 2D Wide.csv", header=None).values
g_2_n = pd.read_csv("Gaussian 2D Narrow.csv", header=None).values
g_2_o = pd.read_csv("Gaussian 2D Overlap.csv", header=None).values
m_2_w = pd.read_csv("Moons 2D Wide.csv", header=None).values
m_2_n = pd.read_csv("Moons 2D Narrow.csv", header=None).values
m_2_o = pd.read_csv("Moons 2D Overlap.csv", header=None).values


g_or_m = input("Choose the type Gaussian or Moon. g = Gaussian, m = Moon :  ")
data_choice = input("Choose which data to use. w = wide, n = narrow, o = overlap :  ")

match g_or_m:
    case "g":
        match data_choice:
            case "w":
                data = g_2_w
            case "n": 
                data = g_2_n
            case "o": 
                data = g_2_o
            case _:
                raise ValueError("Invalid Data choice")
    
    case "m":
        match data_choice:
            case "w":
                data = m_2_w
            case "n":
                data = m_2_n
            case "o":
                data = m_2_o
            case _:
                raise ValueError("Invalid Data choice")
    case _:
        raise ValueError("Invalid Type choice")




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

runs = 30
epochs = 200
accu_values = []
loss_values = []

for seed in range(runs):
    print(f"---------Starting Run : {seed} -----------")
    # Divide into train + validate and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    # Now divide into train and validate
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.25,
        random_state=seed,
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
        epochs=epochs, 
        validation_data=(X_val, y_val)
    )


    # Evaluate the model on the test data
    loss, accu = model.evaluate(X_test, y_test)
    accu_values.append(accu)
    loss_values.append(loss)


test_accs = np.array(accu_values)
test_losses = np.array(loss_values)

print(f"Runs: {runs}")
print(f"Test accuracy: mean={test_accs.mean():.4f}, std={test_accs.std(ddof=1):.4f}, "
      f"min={test_accs.min():.4f}, max={test_accs.max():.4f}")
print(f"Test loss:     mean={test_losses.mean():.4f}, std={test_losses.std(ddof=1):.4f}")

with open(g_or_m + "_2_" + data_choice + ".txt", "w") as f:
    f.write("Experiment Results\n")
    f.write(f"Runs: {runs}\n")
    f.write(f"epochs per run: {epochs}\n")
    f.write(f"Accuracy mean: {test_accs.mean():.4f}\n")
    f.write(f"Accuracy std: {test_accs.std(ddof=1):.4f}\n")
    f.write(f"Loss mean: {test_losses.mean():.4f}\n")
    f.write(f"Loss std: {test_losses.std(ddof=1):.4f}\n")

# what kind of data do we need to report?
# so confusion matrix



# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.freecodecamp.org/news/binary-classification-made-simple-with-tensorflow/
# https://keras.io/guides/training_with_built_in_methods/