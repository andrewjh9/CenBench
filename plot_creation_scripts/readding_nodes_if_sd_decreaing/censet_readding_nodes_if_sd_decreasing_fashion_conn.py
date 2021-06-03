import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_CenSet_conn = np.genfromtxt('results/censet_readding_nodes_if_sd_decreasing/CenSET_laplacian_fashion_mnist_for_200_epochs_20210603-200304_num_sd_2.5_connections__readding_if_sd_is_decreasing_no_scaling_re_adding_all.csv',delimiter='')
read_dataset_Set_conn = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__connections.csv',delimiter='')[0:200]


plt.plot(read_dataset_CenSet_conn, label="CenSet")
plt.plot(read_dataset_Set_conn, label="Set")

plt.legend()
plt.ylabel("Connections")
plt.xlabel("Epoch[#]")
plt.title("CenSET readding when sd decreasing")
plt.show()
