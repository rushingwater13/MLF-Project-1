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


# Read in the data, don't forget to capture the first row
g_3_w = pd.read_csv("Gaussian 3D Wide.csv", header=None).values
g_3_n = pd.read_csv("Gaussian 3D Narrow.csv", header=None).values
g_3_o = pd.read_csv("Gaussian 3D Overlap.csv", header=None).values

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

print(len(X))
print(len(y))
print(X.shape)
print(y.shape)

runs = 30
epochs = 200
accu_values = []
loss_values = []
totalcm = np.zeros((2,2),dtype = int)

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
    # ReLU in the hidden layer because the data set is needs curvature
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
        validation_data=(X_val, y_val)
    )


    # Evaluate the model on the test data
    loss, accu = model.evaluate(X_test, y_test)
    accu_values.append(accu)
    loss_values.append(loss)

    #Keras output for sigmoid, round to match true label
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    conf = confusion_matrix(y_test,y_pred)
    totalcm += conf

test_accs = np.array(accu_values)
test_losses = np.array(loss_values)

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

# what kind of data do we need to report?
# so confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=totalcm, display_labels=["Class 0","Class 1"])
disp.plot(cmap=plt.cm.Blues)

# Show the plot
plt.title("Confusion Matrix")
plt.show()

# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.freecodecamp.org/news/binary-classification-made-simple-with-tensorflow/
# https://keras.io/guides/training_with_built_in_methods/
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html