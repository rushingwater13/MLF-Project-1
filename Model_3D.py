import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# ------------------ Data Wrangling ------------------ #

# Read in the data, don't forget to capture the first row
g_3_w = pd.read_csv("Gaussian 3D Wide.csv", header=None).values
g_3_n = pd.read_csv("Gaussian 3D Narrow.csv", header=None).values
g_3_o = pd.read_csv("Gaussian 3D Overlap.csv", header=None).values


# Capture user input on the specific data set to use
data_choice = input("Choose which data to use. w = wide, n = narrow, o = overlap :  ")

match data_choice:
            case "w":
                data = g_3_w
            case "n": 
                data = g_3_n
            case "o": 
                data = g_3_o
            case _:
                raise ValueError("Invalid Data choice")
      
# Divide the classes
class0 = data[:, 0:3]
class1 = data[:, 3:6]      

# Combine the classes into one array
X = np.vstack((class0, class1))

# Create an array to hold the classifications
y_class0 = np.zeros(len(class0))
y_class1 = np.ones(len(class1))

# Add the classifications to the data array
y = np.concatenate((y_class0, y_class1))

# Shuffle to distribute the data
X, y = shuffle(X, y, random_state=42)



# -------------------------- Begin the Model! -------------------------- #
# We learned how to build this model from the following websites:
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.freecodecamp.org/news/binary-classification-made-simple-with-tensorflow/


runs = 30
epochs = 200
accu_values = []
loss_values = []
totalcm = np.zeros((2,2),dtype = int)

# Repeat the building, training, validating, and testing of the model with a different seed 
# runs numbers of times to confirm the consistency of the model
for seed in range(runs):
    print(f"---------Starting Run : {seed} -----------")
    # Divide into train + validate and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=seed,
        stratify=y
    )

    # Now divide into train and validate
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=1/9,
        random_state=seed,
        stratify=y_temp
    )

    # Set up the model, using tensorflow keras
    # Sequential because Feed Forward
    # ReLU in the hidden layer because the data set needs curvature
    # Sigmoid for the output layer because this is binary classification
    model = models.Sequential([
        layers.Dense(units=16, activation='relu', input_shape=(3,)),
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
        # The following line specifically came from this website: https://keras.io/guides/training_with_built_in_methods/
        validation_data=(X_val, y_val), 
        verbose = 0
    )

    # Evaluate the model on the test data
    loss, accu = model.evaluate(X_test, y_test)
    accu_values.append(accu)
    loss_values.append(loss)

    # Keras output for sigmoid, round to match true label
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    conf = confusion_matrix(y_test,y_pred)
    totalcm += conf


# ----------------- Model Performance Data ----------------- #
test_accs = np.array(accu_values)
test_losses = np.array(loss_values)

# Calculate the averages across the runs and report them
print(f"Runs: {runs}")
print(f"Test accuracy: mean={test_accs.mean():.4f}, std={test_accs.std(ddof=1):.4f}, "
      f"min={test_accs.min():.4f}, max={test_accs.max():.4f}")
print(f"Test loss:     mean={test_losses.mean():.4f}, std={test_losses.std(ddof=1):.4f}")

with open("g" + "_3_" + data_choice + ".txt", "w") as f:
    f.write("Experiment Results\n")
    f.write(f"Runs: {runs}\n")
    f.write(f"epochs per run: {epochs}\n")
    f.write(f"Accuracy mean: {test_accs.mean():.4f}\n")
    f.write(f"Accuracy std: {test_accs.std(ddof=1):.4f}\n")
    f.write(f"Loss mean: {test_losses.mean():.4f}\n")
    f.write(f"Loss std: {test_losses.std(ddof=1):.4f}\n")


# Build the confusion matrix
# We learned the basics of this from the following website:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
disp = ConfusionMatrixDisplay(confusion_matrix=totalcm, display_labels=["Class 0","Class 1"])
disp.plot(cmap=plt.cm.Blues)

# Determine the data set for a nice title on the plots
title_option = ""
if data_choice == "w":
    title_option = "Gaussian 3D Wide" 
elif data_choice == "n":
    title_option = "Gaussian 3D Narrow"
elif data_choice == "o":
    title_option = "Gaussian 3D Overlap"

# Show the plot
plt.title("Confusion Matrix for " + title_option + " Data Set")
plt.savefig("g_" + data_choice + "3D_CM.png")



# ----------------- Decision Boundary Graph ----------------- #
# Graph the points with the decision boundary
# This is using the last repetition of the model
# Using the code found at this website, but extending to 3 dimensions: 
# https://jonchar.net/notebooks/Artificial-Neural-Network-with-Keras/

# Create a mesh grid of points to try to find the boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

# Dropping the grid resolution so that the computation is lighter
x_span = np.linspace(x_min, x_max, 40)
y_span = np.linspace(y_min, y_max, 40)
z_span = np.linspace(z_min, z_max, 40)

xx, yy, zz = np.meshgrid(x_span, y_span, z_span)

grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# Use the model to predict the classifications for the dummy points
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

# Plot the boundary as an approximation of points and the original points
# This part is different from the source, following what we already knew

# Find the points close to the boundary
boundary_points = np.abs(Z - 0.5) < 0.02

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points nearest the boundary as the boundary instead
ax.scatter(xx[boundary_points], yy[boundary_points], zz[boundary_points], 
           s=1, alpha=0.5)

ax.scatter(X[y==0][:, 0], X[y==0][:, 1], X[y==0][:, 2])
ax.scatter(X[y==1][:, 0], X[y==1][:, 1], X[y==1][:, 2])

plt.title("Decision Boundary Approximation for " + title_option + " Data Set")
plt.savefig("g_" + data_choice + "3D_DB.png")


