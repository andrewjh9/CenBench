import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib





dataset_prune_lv_2_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210609-152630_num_sd_2.0_accuracy__narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p1_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-103149_num_sd_2.1_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p2_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-113806_num_sd_2.2_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p3_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-124506_num_sd_2.3000000000000003_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p4_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-135139_num_sd_2.4000000000000004_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p5_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-145851_num_sd_2.5000000000000004_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p6_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-160538_num_sd_2.6000000000000005_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p7_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-171127_num_sd_2.7000000000000006_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p8_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-181713_num_sd_2.8000000000000007_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p9_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210609-170320_num_sd_2.9_accuracy__narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_3_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210609-180920_num_sd_3.0_accuracy__narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')


print(np.mean(dataset_prune_lv_2_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p1_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p2_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p3_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p4_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p5_acc[-10:])*100)

print(np.mean(dataset_prune_lv_2p6_acc[-10:])*100)

print(np.mean(dataset_prune_lv_2p7_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p8_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p9_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3_acc[-10:])*100)

