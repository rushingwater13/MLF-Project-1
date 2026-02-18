import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv("Gaussian 2D Wide.csv")
data = data.values

class1 = data[:, 0:2]
class2 = data[:, 2:4]

stack = np.vstack((class1, class2))