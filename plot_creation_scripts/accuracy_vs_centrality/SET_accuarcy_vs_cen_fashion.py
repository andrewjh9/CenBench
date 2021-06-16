import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from itertools import groupby

axis_label_size = 22
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-2}

plt.rc('font', **font)


read_dataset_cen =  np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_mean_lap__sd_dis_.csv',delimiter='')
read_dataset_cen = [k for k,g in groupby(read_dataset_cen) if k!=0] # Removing duplicated in dataset
print(len(read_dataset_cen))
read_dataset_acc = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_accuracy__sd_dis_.csv',delimiter='')

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_acc*100, '-k')
ax1.set_ylabel('Accuracy [%]', fontsize=axis_label_size)
ax1.set_xlabel("Epoch [#]", fontsize=axis_label_size)


ax2 = ax1.twinx()
ax2.plot(read_dataset_cen, 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r', fontsize=axis_label_size)
ax2.set_xlabel("Epoch [#]")



plt.xlabel("Epoch")
# plt.title("SET Accuracy vs Laplacian Centrality on FashionMNIST")
# plt.show()
plt.grid()

plt.savefig("plots/svg/cen_vs_acc/SET_fashion_cen_acc.svg")