import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_01 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210521-232409_zeta_0.01.csv',delimiter='')
read_dataset_1 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-105413_zeta_0.1.csv',delimiter='')
read_dataset_15 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-115038_zeta_0.15.csv',delimiter='')
read_dataset_2 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-144121_zeta_0.2.csv',delimiter='')
read_dataset_25 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-160419_zeta_0.25.csv',delimiter='')
read_dataset_0001 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-180631_zeta_0.0001.csv',delimiter='')
read_dataset_0 = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-190343_zeta_0.0.csv',delimiter='')




plt.plot(read_dataset_25, label="0.25")
plt.plot(read_dataset_2, label="0.2")
plt.plot(read_dataset_15, label="0.15")
plt.plot(read_dataset_1, label="0.1")
plt.plot(read_dataset_01, label="0.01")
plt.plot(read_dataset_0001, label="0.001")
plt.plot(read_dataset_0, label="0.0")
plt.legend( title="Zeta")
plt.xlabel("Accuracy")
plt.ylabel("Epoch")
tikzplotlib.save("plots/tex/zeta_vs_accuracy_100_epochs.tex")
