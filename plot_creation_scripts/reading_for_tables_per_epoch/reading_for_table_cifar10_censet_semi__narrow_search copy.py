import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib






dataset_prune_lv_2p8_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210611-125825_num_sd_2.8_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')
dataset_prune_lv_3p2_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-180514_num_sd_3.2_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')
dataset_prune_lv_3p6_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-234033_num_sd_3.6_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')
dataset_prune_lv_4_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210613-122653_num_sd_4.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')

print(np.mean(dataset_prune_lv_2_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p4_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p8_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3p2_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3p6_acc[-10:])*100)
print(np.mean(dataset_prune_lv_4_acc[-10:])*100)
