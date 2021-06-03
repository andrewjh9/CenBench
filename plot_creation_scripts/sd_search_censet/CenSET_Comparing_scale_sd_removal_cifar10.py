import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__cifar10_for_300_epochs_20210601-174032_zeta__accuracy_.csv',delimiter='')[0:100]



read_dataset_acc_sd_0p5 =np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-095236_num_sd_0.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_1 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-111429_num_sd_1.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_1p5 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-123540_num_sd_1.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_2 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-135757_num_sd_2.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_2p5 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-151822_num_sd_2.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_3 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-163903_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_3p5 = [0] * 100

read_dataset_acc_sd_4 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-192258_num_sd_4.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_4p5 = np.genfromtxt('results/finding_sd_value_pruning_censet_cifar10/CenSET_laplacian_cifar10_for_100_epochs_20210602-210441_num_sd_4.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')


plt.title("Comapring Accuracy  of CenSET at different removal rates vs SET on Cifar 10")
plt.plot(read_dataset_acc_set, label="SET" )

plt.plot(read_dataset_acc_sd_0p5, label=" Remove node < sd * 0.5" )
plt.plot(read_dataset_acc_sd_1, label=" Remove node < sd * 1" )
plt.plot(read_dataset_acc_sd_1p5, label=" Remove node < sd *  1.5" )
plt.plot(read_dataset_acc_sd_2, label=" Remove node < sd * 2" )
plt.plot(read_dataset_acc_sd_2p5, label=" Remove node < sd * 2.5" )
plt.plot(read_dataset_acc_sd_3, label=" Remove node < sd * 3" )
plt.plot(read_dataset_acc_sd_3p5, label=" MISSING" )
plt.plot(read_dataset_acc_sd_4, label=" Remove node < sd * 4" )
# plt.plot(read_dataset_acc_censet_sd_2p6, label=" Remove node < sd * 2.6")
# plt.plot(read_dataset_acc_censet_sd_2p7, label=" Remove node < sd * 2.7")
# plt.plot(read_dataset_acc_censet_sd_3, label=" Remove node < sd * 3")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()

plt.show()