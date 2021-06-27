import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}
plt.rc('font', **font)


read_dataset_acc_mlp = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_accuracy__mlp_saving_dis_at_multi_25.csv',delimiter='')

read_dataset_acc_censet = np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210618-131730_num_sd_3.7_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_set = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_accuracy__getting_distribution_cifar10_set.csv',delimiter='')

plt.plot(read_dataset_acc_mlp*100 , label ="MLP")
plt.plot(read_dataset_acc_censet*100 , label ="CenSET")
# plt.plot(read_dataset_accs_accset , label ="AccSET")
plt.plot(read_dataset_acc_set*100 , label ="SET")
plt.legend()
plt.xlabel("Epoch[#]", fontsize=axis_label_size)
plt.ylabel("Accuracy [%]", fontsize=axis_label_size)
# plt.title("Different approaches on FashionMNIST")
plt.grid()
# plt.show()
plt.savefig("plots/svg/methods_compare/cifar10_censet_compare.svg")

# tikzplotlib.save("plots/tex/cen_set_compare/fashion_censet_compare.tex")
