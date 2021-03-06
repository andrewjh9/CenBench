import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}
plt.rc('font', **font)


read_dataset_acc_set = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_accuracy__getting_distribution_cifar10_set.csv',delimiter='')



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




# plt.title("Finding optimum $  $ pruning threshold wide search")
plt.plot(read_dataset_acc_set*100, label="SET*" )

plt.plot(dataset_prune_lv_2p8_acc*100, label="$2.8$" )
plt.plot(read_dataset_acc_sd_2p9*100, label="$2.9$" )
plt.plot(read_dataset_acc_sd_3*100, label="$3$" )
plt.plot(read_dataset_acc_sd_3p1*100, label="$3.1$" )
plt.plot(read_dataset_acc_sd_3p2*100, label="$3.2$" )
plt.plot(read_dataset_acc_sd_3p3*100, label="$3.3$" )
plt.plot(read_dataset_acc_sd_3p4*100, label="$3.4$" )
plt.plot(read_dataset_acc_sd_3p5*100, label="$3.5$" )
plt.plot(read_dataset_acc_sd_3p6*100, label="$3.6$" )
plt.plot(read_dataset_acc_sd_3p7*100, label="$3.7$" )
plt.plot(read_dataset_acc_sd_3p8*100, label="$3.8$" )
plt.plot(read_dataset_acc_sd_3p9*100, label="$3.9$" )
plt.plot(read_dataset_acc_sd_4*100, label="$4$" )


plt.xlabel("Epochs [#]", fontsize=axis_label_size)
plt.ylabel("Accuracy [%]", fontsize=axis_label_size )

plt.legend(loc="lower center", ncol=3, prop={'size': 16})
plt.grid()
# plt.show()
plt.savefig("plots/svg/find_opti_sd/find_prune_opti_sigma_narrow_search_cifar10.svg")
# tikzplotlib.save("plots/tex/find_prune_opti_sigma_wide_search_cifar10.tex")