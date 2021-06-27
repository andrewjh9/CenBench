import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}

plt.rc('font', **font)


read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__cifar10_for_300_epochs_20210601-174032_zeta__accuracy_.csv',delimiter='')[0:100]



read_dataset_acc_sd_0p5 =np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-095236_num_sd_0.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_1 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-111429_num_sd_1.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_1p5 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-123540_num_sd_1.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_2 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-135757_num_sd_2.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_2p5 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-151822_num_sd_2.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_3 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-163903_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_3p5 = np.genfromtxt("results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-180045_num_sd_3.5_accuracy_finding_opti_sd_removal_rate.csv",delimiter='') 

read_dataset_acc_sd_4 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-192258_num_sd_4.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_4p5 = np.genfromtxt('results/find_sd_prune_value/cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-210441_num_sd_4.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')


# plt.title("Finding optimum $  $ pruning threshold wide search")
plt.plot(read_dataset_acc_set*100, label="SET*" )

plt.plot(read_dataset_acc_sd_0p5*100, label=" $ 0.5$" )
plt.plot(read_dataset_acc_sd_1*100, label=" $1 $" )
plt.plot(read_dataset_acc_sd_1p5*100, label=" $ 1.5$" )
plt.plot(read_dataset_acc_sd_2*100, label=" $ 2$" )
plt.plot(read_dataset_acc_sd_2p5*100, label=" $ 2.5$" )
plt.plot(read_dataset_acc_sd_3*100, label=" $ 3$" )
plt.plot(read_dataset_acc_sd_3p5*100, label=" $ 3.5$" )
plt.plot(read_dataset_acc_sd_4*100, label=" $ 4$" )


plt.xlabel("Epochs [#]", fontsize=axis_label_size)
plt.ylabel("Accuracy [%]", fontsize=axis_label_size)

plt.legend(loc="lower center", ncol=2,  prop={'size': 16})
plt.grid()
plt.savefig("plots/svg/find_opti_sd/find_prune_opti_sigma_wide_search_cifar10.svg")
# tikzplotlib.save("plots/tex/find_prune_opti_sigma_wide_search_cifar10.tex")