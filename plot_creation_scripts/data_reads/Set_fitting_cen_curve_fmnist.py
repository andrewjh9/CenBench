import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from numpy import arange

from scipy.optimize import curve_fit


# read_dataset_acc = np.genfromtxt('results/SET__fashion_mnist_for_100_epochs_20210529-104007_zeta__accuracy_.csv',delimiter='')
# read_dataset_lap = np.genfromtxt('results/SET__fashion_mnist_for_100_epochs_20210529-104007_zeta__lap.csv',delimiter='')
read_dataset_acc = np.genfromtxt('results/SET__fashion_mnist_for_300_epochs_20210529-123942_zeta__accuracy_.csv',delimiter='')
read_dataset_lap = np.genfromtxt('results/SET__fashion_mnist_for_300_epochs_20210529-123942_zeta__lap.csv',delimiter='')


# https://machinelearningmastery.com/curve-fitting-with-python/


# plt.plot(read_dataset_lap, label="Laplacian centrality")
# plt.plot(read_dataset_acc, label="Accuracy")
# plt.plot(read_dataset_conn, label="# Connections")
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_acc)
ax1.set_ylabel('Accuracy')

# ax1 = fig.add_subplot(111)
# ax1.plot(read_dataset_conn)
# ax1.set_ylabel('# Connections')

ax2 = ax1.twinx()
ax2.plot(read_dataset_lap, 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r')


plt.xlabel("Epoch")
plt.title("Accuracy vs Laplacian Centrality on FashionMnist")
plt.show()

# tikzplotlib.save("plots/tex/lap_vs_accuracy_300_epochs.tex")



x = range(0, 300)
y = read_dataset_lap

# define the true objective function
def objective(x, a, b):
	return a * x + b
 

# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
print('y = %.5f * x + %.5f' % (a, b))
# plot input vs output
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.show()