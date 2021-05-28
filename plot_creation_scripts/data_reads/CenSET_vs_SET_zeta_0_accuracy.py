import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_set_0 = np.genfromtxt('results/SET_accuracy_cifar10_for_100_epochs_20210527-171629_accuracy_zeta_0.csv',delimiter='')
read_dataset_censet_0 = np.genfromtxt('results/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210527-183707_accuracy_zeta_0.csv',delimiter='')
read_dataset_censet_05 = np.genfromtxt('results/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210527-195604_accuracy_zeta_0.05.csv',delimiter='')

plt.plot(read_dataset_set_0, label="SET (zeta: 0)")
plt.plot(read_dataset_censet_0, label="CenSET (zeta: 0)")
plt.plot(read_dataset_censet_05, label="CenSET (zeta: 0.05)")


plt.legend( title="Method")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
plt.title("Accuracy across epochs of different methods with Zeta = 0")
plt.show()