import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_CenSet_acc = np.genfromtxt('results/censet_readding_nodes_if_sd_decreasing/CenSET_laplacian_fashion_mnist_for_200_epochs_20210603-200304_num_sd_2.5_accuracy__readding_if_sd_is_decreasing_no_scaling_re_adding_all.csv',delimiter='')
read_dataset_Set_acc = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:200]


plt.plot(read_dataset_CenSet_acc, label="CenSet Acc")
plt.plot(read_dataset_Set_acc, label="Set Acc")

plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Epoch[#]")
plt.title("CenSET readding when sd decreasing")
plt.show()
