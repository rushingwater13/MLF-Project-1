#import matplotlib
#matplotlib.use("Agg")  # Use non-GUI backend

import pandas as pd
import matplotlib.pyplot as plt

# Load the csv file
data = pd.read_csv('Gaussian 2D Wide.csv', header=None)

# Extract columns
class1_x = data.iloc[:, 0]
class1_y = data.iloc[:, 1]
class2_x = data.iloc[:, 2]
class2_y = data.iloc[:, 3]

# Create scatter plot
plt.figure()
plt.scatter(class1_x, class1_y, marker='o', label='Class 1')
plt.scatter(class2_x, class2_y, marker='x', label='Class 2')

# Labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Class 1 vs Class 2 Data')
plt.legend()

plt.savefig("Gaussian 2D Wide.png")
print("Plot saved as plot.png")

