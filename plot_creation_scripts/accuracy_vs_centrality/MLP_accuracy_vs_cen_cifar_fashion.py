import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from itertools import groupby

fig, axes = plt.subplots(1,2, figsize=(8,8))


read_dataset_cen_fashion =  np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_mean_lap__base_line.csv',delimiter='')
read_dataset_cen_fashion = [k for k,g in groupby(read_dataset_cen_fashion) if k!=0] # Removing duplicated in dataset
read_dataset_acc_fashion = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_accuracy__base_line.csv',delimiter='')


axes[1][1] = fig.add_subplot(111)
axes[1][1] = plot(read_dataset_acc_fashion*100, 'k-')
axes[1][1] .set_ylabel('Accuracy [%]')
axes[1][1] .set_xlabel("Epoch [#]")


axes_0_twin = axes[1][1] .twinx()
axes_0_twin.plot(read_dataset_cen_fashion, 'r-')
axes_0_twin.set_ylabel("Laplacian Centrality", color='r')
axes_0_twin.set_xlabel("Epoch [#]")

read_dataset_cen_cifar =  np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_mean_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_cen_cifar = [k for k,g in groupby(read_dataset_cen_cifar) if k!=0] # Removing duplicated in dataset
read_dataset_acc_cifar = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_accuracy__mlp_saving_dis_at_multi_25.csv',delimiter='')

fig = plt.figure()

axes[0][1] = fig.add_subplot(111)
axes[0][1].plot(read_dataset_acc_cifar*100)
axes[0][1].set_ylabel('Accuracy [%]')
axes[0][1].set_xlabel("Epoch [#]")


ax2 = axes[0][1].twinx()
ax2.plot(read_dataset_cen_cifar, 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r')
ax2.set_xlabel("Epoch [#]")


plt.tight_layout()
plt.show()

# tikzplotlib.save("plots/tex/cen_vs_acc/MLP_fashion_cen_vs_acc.tex")