import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_acc_set = np.genfromtxt('results/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__accuracy_.csv',delimiter='')[0:300]

read_dataset_acc_censet_sd_2p6 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-100124_num_sd_2.6_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_2p7 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-130737_num_sd_2.7_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_acc_censet_sd_3 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-070411_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')



# read_dataset_acc_sd_0 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-180409_num_sd_0.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]
# read_dataset_acc_sd_0p5 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-182001_num_sd_0.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]
# read_dataset_acc_sd_1 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-183523_num_sd_1.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]
# read_dataset_acc_sd_1p5= np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-185215_num_sd_1.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]
# read_dataset_acc_sd_2 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-190845_num_sd_2.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]
# read_dataset_acc_sd_2p5 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-192526_num_sd_2.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]

# read_dataset_acc_sd_3 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-194333_num_sd_3.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]

# read_dataset_acc_sd_3p5 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-195937_num_sd_3.5_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]

# read_dataset_acc_sd_4 = np.genfromtxt('results/finding_sd_value_pruning/CenSET_laplacian_fashion_mnist_for_50_epochs_20210601-201842_num_sd_4.0_accuracy_finding_opti_sd_removal_rate.csv',delimiter='')[10:50]



plt.title("Comapring Accuracy at different removal rates epochs = 300 - Starting at 10")
plt.plot(read_dataset_acc_set, label="SET" )

# plt.plot(read_dataset_acc_sd_0, label=" Remove node < sd * 0" )
# plt.plot(read_dataset_acc_sd_0p5, label=" Remove node < sd * 0.5" )
# plt.plot(read_dataset_acc_sd_1, label=" Remove node < sd * 1" )
# plt.plot(read_dataset_acc_sd_1p5, label=" Remove node < sd *  1.5" )
# plt.plot(read_dataset_acc_sd_2, label=" Remove node < sd * 2" )
# plt.plot(read_dataset_acc_sd_2p5, label=" Remove node < sd * 2.5" )
# plt.plot(read_dataset_acc_sd_3, label=" Remove node < sd * 3" )
# plt.plot(read_dataset_acc_sd_3p5, label=" Remove node < sd * 3.5" )
# plt.plot(read_dataset_acc_sd_4, label=" Remove node < sd * 4" )
plt.plot(read_dataset_acc_censet_sd_2p6, label=" Remove node < sd * 2.6")
plt.plot(read_dataset_acc_censet_sd_2p7, label=" Remove node < sd * 2.7")
plt.plot(read_dataset_acc_censet_sd_3, label=" Remove node < sd * 3")

plt.legend()

plt.show()