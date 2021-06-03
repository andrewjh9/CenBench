import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_conn_censet_no_readd_sd_2p5 = np.genfromtxt('results/censet_not_readding_connections/epochs300/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-212937_num_sd_2.5_connections_Not_readding_connections.csv',delimiter='')
# read_dataset_conn_censet_sd_no_readd_= np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-160022_num_sd_1.5_connections_Not_readding_connections.csv',delimiter='')
# read_dataset_conn_censet_sd_no_readd_ = np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-161638_num_sd_2.0_connections_Not_readding_connections.csv',delimiter='')
# read_dataset_conn_cense_no_readd_sd_ = np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-163250_num_sd_2.5_connections_Not_readding_connections.csv',delimiter='')


read_dataset_conn_set = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__connections.csv',delimiter='')[0:300]


plt.title("Comapring Connections SET vs CenSET (not readding connections)")
plt.plot(read_dataset_conn_set, label="SET" )
plt.plot(read_dataset_conn_censet_no_readd_sd_2p5, label="CenSET Remove cen(node) < mean -  sd * 2.5 Not readding connections" )
plt.xlabel("Epochs")
plt.ylabel("Connections")

# plt.plot(read_dataset_conn_censet_sd_1p5, label=" Remove node < sd *  1.5" )
# plt.plot(read_dataset_conn_censet_sd_2, label=" Remove node < sd * 2" )
# plt.plot(read_dataset_conn_censet_sd_2p5, label=" Remove node < sd * 2.5" )

plt.legend()

plt.show()