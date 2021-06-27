import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}

plt.rc('font', **font)


read_dataset_acc_set = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_accuracy__getting_distribution_cifar10_set.csv',delimiter='')


read_dataset_acc_sd_2 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-153741_num_sd_2.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_2p4 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-211454_num_sd_2.4_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_2p8 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210611-125825_num_sd_2.8_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_3p2 = np.genfromtxt("results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-180514_num_sd_3.2_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv",delimiter='') 

read_dataset_acc_sd_3p6 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-234033_num_sd_3.6_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_4 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210613-122653_num_sd_4.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')


# plt.title("Finding optimum $  $ pruning threshold wide search")
plt.plot(read_dataset_acc_set*100, label="SET*" )

plt.plot(read_dataset_acc_sd_2*100, label="  $ 2$" )
plt.plot(read_dataset_acc_sd_2p4*100, label="  $ 2.4$" )
plt.plot(read_dataset_acc_sd_2p8*100, label="$ 2.8$" )
plt.plot(read_dataset_acc_sd_3p2*100, label=" $ 3.2$" )
plt.plot(read_dataset_acc_sd_3p6*100, label=" $ 3.6$" )
plt.plot(read_dataset_acc_sd_4*100, label=" $ 4$" )


plt.xlabel("Epochs [#]", fontsize=axis_label_size)
plt.ylabel("Accuracy [%]", fontsize=axis_label_size)
plt.grid()
plt.legend( loc="lower center", ncol=2 , prop={'size': 16}   )
# plt.show()
plt.savefig("plots/svg/find_opti_sd/find_prune_opti_sigma_semi_narrow_search_cifar10.svg")