import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:300]

read_dataset_acc_censet_no_readd_sd_2p5 = np.genfromtxt('results/censet_not_readding_connections/epochs300/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-212937_num_sd_2.5_accuracy_Not_readding_connections.csv',delimiter='')

read_dataset_acc_censet__sd_2p6 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet__sd_2p7 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-130737_num_sd_2.7_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
# read_dataset_acc_censet_no_readd_sd_ = np.genfromtxt('',delimiter='')
# read_dataset_acc_censet_no_readd_sd_ = np.genfromtxt('',delimiter='')
# read_dataset_acc_censet_no_readd_sd_ = np.genfromtxt('',delimiter='')




plt.title("Comapring Accuracy SET vs CenSET (not readding connections)")
plt.plot(read_dataset_acc_set, label="SET" )
plt.plot(read_dataset_acc_censet_no_readd_sd_2p5, label="CenSET Remove cen(node)  <  mean - sd * 2.5" )
plt.plot(read_dataset_acc_censet__sd_2p6, label="CenSET Remove cen(node)  <  mean - sd * 2.6 (readding)" )
plt.plot(read_dataset_acc_censet__sd_2p7, label="CenSET Remove cen(node)  <  mean - sd * 2.7 (readding)" )
plt.ylabel("Accuracy")
plt.xlabel("Epochs")

# plt.plot(read_dataset_acc_censet_sd_1p5, label=" Remove node < sd *  1.5" )
# plt.plot(read_dataset_acc_censet_sd_2, label=" Remove node < sd * 2" )
# plt.plot(read_dataset_acc_censet_sd_2p5, label=" Remove node < sd * 2.5" )

plt.legend()

plt.show()