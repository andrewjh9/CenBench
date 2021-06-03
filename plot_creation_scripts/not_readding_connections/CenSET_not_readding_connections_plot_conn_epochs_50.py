import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_conn_censet_sd_1_no_readd = np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-154525_num_sd_1.0_connections_Not_readding_connections.csv',delimiter='')
read_dataset_conn_censet_sd_1p5 = np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-160022_num_sd_1.5_connections_Not_readding_connections.csv',delimiter='')
read_dataset_conn_censet_sd_2 = np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-161638_num_sd_2.0_connections_Not_readding_connections.csv',delimiter='')
read_dataset_conn_censet_sd_2p5 = np.genfromtxt('results/censet_not_readding_connections/CenSET_laplacian_fashion_mnist_for_50_epochs_20210602-163250_num_sd_2.5_connections_Not_readding_connections.csv',delimiter='')


read_dataset_accuracy_set = np.genfromtxt('results/SET__fashion_mnist_for_300_epochs_20210529-123942_zeta__connections.csv',delimiter='')[0:50]


plt.title("Comapring Accuracy at different removal rates epochs = 100 - Not readding connections !")
plt.plot(read_dataset_accuracy_set, label="SET" )

plt.plot(read_dataset_conn_censet_sd_1_no_readd, label=" Remove node < sd * 1" )
plt.plot(read_dataset_conn_censet_sd_1p5, label=" Remove node < sd *  1.5" )
plt.plot(read_dataset_conn_censet_sd_2, label=" Remove node < sd * 2" )
plt.plot(read_dataset_conn_censet_sd_2p5, label=" Remove node < sd * 2.5" )

plt.legend()

plt.show()