import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__cifar10_for_300_epochs_20210601-174032_zeta__accuracy_.csv',delimiter='')[0:100]


read_dataset_acc_sd_2 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-153741_num_sd_2.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_sd_2p4 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210610-211454_num_sd_2.4_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_2p8 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210611-125825_num_sd_2.8_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_3p2 = np.genfromtxt("results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-180514_num_sd_3.2_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv",delimiter='') 

read_dataset_acc_sd_3p6 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210612-234033_num_sd_3.6_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_sd_4 = np.genfromtxt('results/find_sd_prune_value/cifar10/semi_narrow/CenSET_laplacian_cifar10_for_400_epochs_20210613-122653_num_sd_4.0_accuracy__semi_narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')


plt.title("Finding optimum $ \sigma $ pruning threshold wide search")
plt.plot(read_dataset_acc_set*100, label="SET (baseline)" )

# plt.plot(read_dataset_acc_sd_2*100, label=" LC(k) < $\mu - 2\sigma$" )
# plt.plot(read_dataset_acc_sd_2p4*100, label=" LC(k) < $\mu - 2,4\sigma$" )
plt.plot(read_dataset_acc_sd_2p8*100, label=" LC(k) < $\mu - 2.8\sigma$" )
plt.plot(read_dataset_acc_sd_3p2*100, label=" LC(k) < $\mu - 3.2\sigma$" )
plt.plot(read_dataset_acc_sd_3p6*100, label=" LC(k) < $\mu - 3.6\sigma$" )
plt.plot(read_dataset_acc_sd_4*100, label=" LC(k) < $\mu - 4\sigma$" )


plt.xlabel("Epochs [#]")
plt.ylabel("Accuracy [%]")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# tikzplotlib.save("plots/tex/find_prune_opti_sigma_wide_search_cifar10.tex")