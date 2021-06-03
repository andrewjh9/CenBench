import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_CenSet_sd = np.genfromtxt('results/censet_plotting_sd/CenSET_laplacian_fashion_mnist_for_200_epochs_20210603-170025_num_sd_2.5_sd_lap__seeing_censset_sd_change_test.csv',delimiter='')

read_dataset_CenSet_sd = np.genfromtxt('results/censet_plotting_sd/CenSET_laplacian_fashion_mnist_for_200_epochs_20210603-170025_num_sd_2.5_sd_lap__seeing_censset_sd_change_test.csv',delimiter='')
read_dataset_CenSet_mean = np.genfromtxt('results/censet_plotting_sd/CenSET_laplacian_fashion_mnist_for_200_epochs_20210603-170025_num_sd_2.5_mean_lap__seeing_censset_sd_change_test.csv',delimiter='')

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_CenSet_mean)
ax1.set_ylabel('Mean')

# ax1 = fig.add_subplot(111)
# ax1.plot(read_dataset_conn)
# ax1.set_ylabel('# Connections')

ax2 = ax1.twinx()
ax2.plot(read_dataset_CenSet_sd, 'r-')
ax2.set_ylabel("Laplacian Centrality SD", color='r')

plt.legend()

plt.xlabel("Epoch")
plt.title("Accuracy vs SD cen")
plt.show()
