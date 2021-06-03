import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from numpy import arange

from scipy.optimize import curve_fit


# read_dataset_acc = np.genfromtxt('results/SET__fashion_mnist_for_100_epochs_20210529-104007_zeta__accuracy_.csv',delimiter='')
# read_dataset_acc = np.genfromtxt('results/SET__fashion_mnist_for_300_epochs_20210529-123942_zeta__accuracy_.csv',delimiter='')
# read_dataset_lap_fmnist300 = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__connections.csv',delimiter='')[0:1000]
# read_dataset_lap_cifar100 = np.genfromtxt('results/SET__cifar100_for_100_epochs_20210529-140724_zeta__lap.csv',delimiter='')
read_dataset_acc_fmnist_1000 = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')
read_dataset_lap_fmnist_1000 = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__lap.csv',delimiter='')
# read_dataset__cen_set_lap_acc_fmnist_100 = np.genfromtxt('results/CenSET_laplacian_fashion_mnist_for_100_epochs_20210601-141055_zeta__accuracy_.csv', delimiter='')
# read_dataset__cen_set_lap_acc_fmnist_100_3_sds = np.genfromtxt('results/CenSET_laplacian_fashion_mnist_for_100_epochs_20210601-164158_zeta_3_sds_accuracy_.csv', delimiter='')
# https://machinelearningmastery.com/curve-fitting-with-python/

# read_dataset_lap_cifar10_100 = results/SET__cifar10_for_300_epochs_20210601-174032_zeta__lap.csv
# read_dataset_acc_cifar10_100 = results/SET__cifar10_for_300_epochs_20210601-174032_zeta__accuracy_.csv
# plt.plot(read_dataset_lap_fmnist300)
# plt.plot(read_dataset__cen_set_lap_acc_fmnist_100, label="cen_fmnist" )
# plt.plot(read_dataset_acc[0:100], label="set_fmnist" )
# plt.plot(read_dataset__cen_set_lap_acc_fmnist_100_3_sds, label="cen_fmnist remove 3 sds")
# # plt.plot(read_dataset_lap_fmnist100, label="fmnist 100 Real")
# # plt.plot(read_dataset_lap_fmnist_1000[0:100], label="fmnist 100 fake")
# # plt.plot(read_dataset_lap_cifar100, label="Cifar10 100")
# # plt.plot(read_dataset_lap_fmnist_1000, label="fmnist 1000")
# # plt.plot(read_dataset_lap_fmnist300, label="fminst 300")
# plt.legend()

# plt.show()


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_acc_fmnist_1000[0:1000])
ax1.set_ylabel('Accuracy')

# ax1 = fig.add_subplot(111)
# ax1.plot(read_dataset_conn)
# ax1.set_ylabel('# Connections')

ax2 = ax1.twinx()
ax2.plot(read_dataset_lap_fmnist_1000[0:1000], 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r')


plt.xlabel("Epoch")
plt.title("SET Accuracy vs Laplacian Centrality on FashionMnist")
plt.show()

# tikzplotlib.save("plts/tex/lap_vs_accuracy_300_epochs.tex")



x = range(0, 1000)
# y = [float(i)/max(read_dataset_lap_fmnist_1000[0:1000])*100 for i in read_dataset_lap_fmnist_1000[0:1000]]
# y = read_dataset_lap_fmnist_1000
y = read_dataset_lap_fmnist_1000

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