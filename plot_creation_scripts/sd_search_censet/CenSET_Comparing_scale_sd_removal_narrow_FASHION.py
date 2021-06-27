import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}

plt.rc('font', **font)


read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p1 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-103149_num_sd_2.1_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p2 =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-113806_num_sd_2.2_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p3 =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-124506_num_sd_2.3000000000000003_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p4 =  np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-135139_num_sd_2.4000000000000004_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p5 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p6 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p7 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-130737_num_sd_2.7_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]
read_dataset_acc_censet_sd_2p8 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-181713_num_sd_2.8000000000000007_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
# read_dataset_acc_censet_sd_2p9 = np.genfromtxt('',delimiter='')
read_dataset_acc_censet_sd_3 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-070411_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:200]




# plt.title("Finding optimum $  $ pruning threshold wide search MISSING 2.9 and 2")
plt.plot(read_dataset_acc_set*100, label="SET *" )

plt.plot(read_dataset_acc_censet_sd_2p1* 100, label=" $ 2.1$")
plt.plot(read_dataset_acc_censet_sd_2p2* 100, label=" $ 2.2$")
plt.plot(read_dataset_acc_censet_sd_2p3* 100, label=" $ 2.3$")
plt.plot(read_dataset_acc_censet_sd_2p4* 100, label=" $ 2.4$")
plt.plot(read_dataset_acc_censet_sd_2p5* 100, label=" $ 2.5$")
plt.plot(read_dataset_acc_censet_sd_2p6*100, label=" $ 2.6$")
plt.plot(read_dataset_acc_censet_sd_2p7*100, label=" $ 2.7$")
plt.plot(read_dataset_acc_censet_sd_2p8*100, label=" $ 2.8$")
plt.plot(read_dataset_acc_censet_sd_3*100, label=" $ 3$")

plt.xlabel("Epochs [#]", fontsize=axis_label_size)
plt.ylabel("Accuracy [%] ", fontsize=axis_label_size)
plt.grid()
plt.legend(  loc="lower center", ncol=2)
# plt.show()
plt.savefig("plots/svg/find_opti_sd/find_prune_opti_sigma_narrow_search_fashion.svg")
# tikzplotlib.save("plots/tex/find_prune_opti_sigma_narrow_search_fashion.tex")