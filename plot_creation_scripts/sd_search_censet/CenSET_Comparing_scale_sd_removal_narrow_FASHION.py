import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_acc_set = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:50]

# read_dataset_acc_censet_sd_2p6 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p6 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p7 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-130737_num_sd_2.7_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_3 = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-070411_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')




plt.title("Finding optimum $ \sigma $ pruning threshold wide search MISSING 2 - > 2.5")
plt.plot(read_dataset_acc_set*100, label="SET (Baseline)" )


# plt.plot(read_dataset_acc_censet_sd_2p5* 100, label="Prune node k if LC(k) < $\mu - 2.5\sigma$")
plt.plot(read_dataset_acc_censet_sd_2p6*100, label="Prune node k if LC(k) < $\mu - 2.6\sigma$")
plt.plot(read_dataset_acc_censet_sd_2p7*100, label="Prune node k if LC(k) < $\mu - 2.7\sigma$")
plt.plot(read_dataset_acc_censet_sd_3*100, label="Prune node k if LC(k) < $\mu - 3\sigma$")

plt.xlabel("Epochs [#]")
plt.ylabel("Accuracy [%] ")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()
tikzplotlib.save("plots/tex/find_prune_opti_sigma_narrow_search_fashion.tex")