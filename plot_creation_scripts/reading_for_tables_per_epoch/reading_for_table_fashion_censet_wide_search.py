import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


dataset_prune_lv_0_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-180409_num_sd_0.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_0p5_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-180409_num_sd_0.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_1_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-183523_num_sd_1.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_1p5_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-185215_num_sd_1.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-190845_num_sd_2.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_2p5_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-192526_num_sd_2.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_3_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-194333_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_3p5_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-195937_num_sd_3.5_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

dataset_prune_lv_4_acc =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-201842_num_sd_4.0_accuracy_finding_opti_sd_removal_rate.csv', delimiter='')

print(np.mean(dataset_prune_lv_0_acc[-10:])*100)
print(np.mean(dataset_prune_lv_0p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_1_acc[-10:])*100)
print(np.mean(dataset_prune_lv_1p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2_acc[-10:])*100)
print(np.mean(dataset_prune_lv_2p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3_acc[-10:])*100)
print(np.mean(dataset_prune_lv_3p5_acc[-10:])*100)
print(np.mean(dataset_prune_lv_4_acc[-10:])*100)
