import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib





dataset_prune_lv_0_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210614-180130_num_sd_0_accuracy__wide_search_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_0p5_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-095236_num_sd_0.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_1_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-111429_num_sd_1.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_1p5_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-123540_num_sd_1.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-135757_num_sd_2.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p5_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-151822_num_sd_2.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_3_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-163903_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_3p5_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-180045_num_sd_3.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_4_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-192258_num_sd_4.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_4p5_acc =  np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-210441_num_sd_4.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

print(np.mean(dataset_prune_lv_0_acc[-10:])*100)
print(np.mean(dataset_prune_lv_0p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_1_acc[-10:])*100)
print(np.mean(dataset_prune_lv_1p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_4_acc[-10:])*100)