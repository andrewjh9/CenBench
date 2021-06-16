import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:50]

# read_dataset_acc_censet_sd_2p6 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
# read_dataset_acc_censet_sd_2p7 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-130737_num_sd_2.7_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
# read_dataset_acc_censet_sd_3 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-070411_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')



read_dataset_acc_sd_0 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-180409_num_sd_0.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]
read_dataset_acc_sd_0p5 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-182001_num_sd_0.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]
read_dataset_acc_sd_1 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-183523_num_sd_1.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]
read_dataset_acc_sd_1p5= np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-185215_num_sd_1.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]
read_dataset_acc_sd_2 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-190845_num_sd_2.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]
read_dataset_acc_sd_2p5 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-192526_num_sd_2.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]

read_dataset_acc_sd_3 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-194333_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]

read_dataset_acc_sd_3p5 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-195937_num_sd_3.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]

read_dataset_acc_sd_4 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-201842_num_sd_4.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[0:50]


# plt.title("Finding optimum $ \sigma $ pruning threshold wide search")
plt.plot(read_dataset_acc_set*100, label="SET (Baseline)" )

plt.plot(read_dataset_acc_sd_0 * 100, label=" LC(k) < $\mu - 0\sigma$" )
plt.plot(read_dataset_acc_sd_0p5* 100, label=" LC(k) < $\mu - 0.5\sigma$" )
plt.plot(read_dataset_acc_sd_1* 100, label=" LC(k) < $\mu - \sigma$"  )
plt.plot(read_dataset_acc_sd_1p5* 100, label=" LC(k) < $\mu - 1.5\sigma$"  )
plt.plot(read_dataset_acc_sd_2* 100, label=" LC(k) < $\mu - 2\sigma$"  )
plt.plot(read_dataset_acc_sd_2p5* 100, label=" LC(k) < $\mu - 2.5\sigma$")
plt.plot(read_dataset_acc_sd_3* 100, label=" LC(k) < $\mu - 3\sigma$" )
plt.plot(read_dataset_acc_sd_3p5* 100, label=" LC(k) < $\mu - 3.5\sigma$" )
plt.plot(read_dataset_acc_sd_4* 100, label=" LC(k) < $\mu - 4\sigma$" )
# plt.plot(read_dataset_acc_censet_sd_2p6, label=" Remove node < sd * 2.6")
# plt.plot(read_dataset_acc_censet_sd_2p7, label=" Remove node < sd * 2.7")
# plt.plot(read_dataset_acc_censet_sd_3, label=" Remove node < sd * 3")

plt.xlabel("Epochs [#]")
plt.ylabel("Accuracy [%] ")

plt.legend(title="Prune node k if",bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()
plt.savefig("plots/svg/find_opti_sd/find_prune_opti_sigma_wide_search_fashion.svg")