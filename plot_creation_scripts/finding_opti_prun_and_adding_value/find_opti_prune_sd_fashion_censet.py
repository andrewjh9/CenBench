import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p1 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-103149_num_sd_2.1_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p2 =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-113806_num_sd_2.2_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p3 =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-124506_num_sd_2.3000000000000003_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p4 =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-135139_num_sd_2.4000000000000004_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p5 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p6 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p7 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-130737_num_sd_2.7_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p8 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-181713_num_sd_2.8000000000000007_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
# read_dataset_acc_censet_sd_2p9 = np.genfromtxt('',delimiter='')
read_dataset_acc_censet_sd_3 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-070411_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]

print("2p1", np.average(read_dataset_acc_censet_sd_2p1))
print("2p2", np.average(read_dataset_acc_censet_sd_2p2))
print("2p3", np.average(read_dataset_acc_censet_sd_2p3))
print("2p4", np.average(read_dataset_acc_censet_sd_2p4))
print("2p5", np.average(read_dataset_acc_censet_sd_2p5))
print("2p6", np.average(read_dataset_acc_censet_sd_2p6))
print("2p7", np.average(read_dataset_acc_censet_sd_2p7))
print("2p8", np.average(read_dataset_acc_censet_sd_2p8))

