import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib





dataset_prune_lv_2_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-153741_num_sd_2.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')
dataset_prune_lv_2p4_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-211454_num_sd_2.4_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')
dataset_prune_lv_2p8_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210611-125825_num_sd_2.8_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv', delimiter='')
read_dataset_acc_sd_2p9 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210615-193058_num_sd_2.9_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210616-141759_num_sd_3.0_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p1 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210616-211823_num_sd_3.1_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p2 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-024647_num_sd_3.2_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p3 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-082042_num_sd_3.3000000000000003_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p4 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-135335_num_sd_3.4000000000000004_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p5 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210618-014413_num_sd_3.5_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p6 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210618-072926_num_sd_3.6_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p7 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210618-131730_num_sd_3.7_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p8 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210618-191019_num_sd_3.8000000000000003_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_3p9 =np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210619-115238_num_sd_3.9_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_4 =np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210613-122653_num_sd_4.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

print("2.8: ",np.mean(dataset_prune_lv_2p8_acc[-10:])*100)
print("2.9: ",np.mean(read_dataset_acc_sd_2p9[-10:])*100)
print("3: ",np.mean(read_dataset_acc_sd_3[-10:])*100)
print("3.1: ",np.mean(read_dataset_acc_sd_3p1[-10:])*100)
print("3.2: ", np.mean(read_dataset_acc_sd_3p2[-10:])*100)
print("3.3: ",np.mean(read_dataset_acc_sd_3p3[-10:])*100)
print("3.4: ",np.mean(read_dataset_acc_sd_3p4[-10:])*100)
print("3.5: ",np.mean(read_dataset_acc_sd_3p5[-10:])*100)
print("3.6: ",np.mean(read_dataset_acc_sd_3p6[-10:])*100)
print("3.7: ",np.mean(read_dataset_acc_sd_3p7[-10:])*100)
print("3.8: ",np.mean(read_dataset_acc_sd_3p8[-10:])*100)
print("3.9: ",np.mean(read_dataset_acc_sd_3p9[-10:])*100)
print("4: ",np.mean(read_dataset_acc_sd_4[-10:])*100)
