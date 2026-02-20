import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


g_2_w = pd.read_csv("Gaussian 2D Wide.csv", header=None)
g_2_n = pd.read_csv("Gaussian 2D Narrow.csv", header=None)
g_2_o = pd.read_csv("Gaussian 2D Overlap.csv", header=None)

data_choice = input("Choose which data to use. w = wide, n = narrow, o = overlap :  ")
learning_rate = float(input("Enter the learning rate (0-1) : "))
bias = float(input("Enter the bias : "))


match data_choice:
    case "w":
        data = g_2_w
    case "n": 
        data = g_2_n
    case "o": 
        data = g_2_o
    case _:
        raise ValueError("Invalid choice")


def net(w, p):
    return w[0]*bias + w[1]*p[0][0] + w[2]*p[0][1]

# testdata = [
#     ((-0.1,-1), 0),
#     ((0.2,-0.9),0),
#     ((0.6,0.8), 1),
#     ((0.0,0.0),1),
# ]


# Divide the classes
class0 = data.iloc[:, [0,1]].values
class1 = data.iloc[:, [2,3]].values

#label points based on class : (1.2,2.4), 1
#then combine into one array
xlabeled = []
for point in class0:
    xlabeled.append((point, 0))
for point in class1:
    xlabeled.append((point, 1))


w_v = [0.5,1,-0.3]
epoch = 0
changed = True

print(f"original weight = {w_v}")
while (changed and epoch < 50000):
    epoch += 1
    changed = False
    for p in xlabeled:
        #calculate net value
        net_i = net(w_v,p)
        #using step function assign actual value
        if net_i < 0: a = 0
        else: a = 1
        #get class from label
        target = p[1]
        #if t-a not 0 then get w_new
        if target - a != 0:
            changed = True
            w_v = np.array(w_v) + np.array(list((map(lambda x: x * (target-a) * learning_rate, [bias, p[0][0],p[0][1]]))))

print(f"epoch = {epoch-1}")
print(f"final weight vector = {w_v}")

# Extract columns
class1_x = data.iloc[:, 0]
class1_y = data.iloc[:, 1]
class2_x = data.iloc[:, 2]
class2_y = data.iloc[:, 3]

# Create scatter plot
plt.figure()
plt.scatter(class1_x, class1_y, marker='o', label='Class 1')
plt.scatter(class2_x, class2_y, marker='x', label='Class 2')

# weights: [bias, w1, w2]
w0, w1, w2 = w_v

# choose x range based on data
x_min = min(class0[:,0].min(), class1[:,0].min())
x_max = max(class0[:,0].max(), class1[:,0].max())
x_line = np.linspace(x_min, x_max, 200)

# boundary line: w0 + w1*x + w2*y = 0  ->  y = -(w0 + w1*x)/w2
if abs(w2) < 1e-12:
    # vertical boundary: w0 + w1*x = 0 -> x = -w0/w1
    x_boundary = -w0 / w1
    plt.axvline(x_boundary)
else:
    y_line = -(w0 + w1 * x_line) / w2
    plt.plot(x_line, y_line)

# Labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Class 1 vs Class 2 Data')
plt.legend()

plt.savefig(data_choice + ".png")
print("Plot saved as : " + data_choice + ".png")

