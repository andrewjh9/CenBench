import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_acc_set = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_accuracy__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_acc_censet_sd_2 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-153741_num_sd_2.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p4 =  np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-211454_num_sd_2.4_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p8 =  np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210611-125825_num_sd_2.8_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_3p2 =  np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-180514_num_sd_3.2_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_3p6 =  np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-234033_num_sd_3.6_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_4 =  np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210613-122653_num_sd_4.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

plt.plot(read_dataset_acc_set , label ="SET (baseline)")
plt.plot(read_dataset_acc_censet_sd_2 , label ="LC(k) < $\mu - 2\sigma$")
plt.plot(read_dataset_acc_censet_sd_2p4 , label ="LC(k) < $\mu - 2.4\sigma$")
plt.plot(read_dataset_acc_censet_sd_2p8 , label ="LC(k) < $\mu - 2.8\sigma$")
plt.plot(read_dataset_acc_censet_sd_3p2 , label ="LC(k) < $\mu - 3.2\sigma$")
plt.plot(read_dataset_acc_censet_sd_3p6 , label ="LC(k) < $\mu - 3.6\sigma$")
plt.plot(read_dataset_acc_censet_sd_4 , label ="LC(k) < $\mu - 4\sigma$")



plt.legend(title="Prune node k if ",bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Epoch[\#]")
plt.ylabel("Accuracy [\%]")
# plt.title("Different levels")

plt.savefig("plots/svg/find_opti_sd/cifar10_semi_narrow.svg")

# plt.show()

