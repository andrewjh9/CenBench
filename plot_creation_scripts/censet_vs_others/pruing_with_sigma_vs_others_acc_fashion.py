import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}

plt.rc('font', **font)
read_dataset_acc_mlp = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_accuracy__base_line.csv',delimiter='')

read_dataset_acc_censet = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-171127_num_sd_2.7000000000000006_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_set = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_accuracy__sd_dis_.csv',delimiter='')

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
plt.savefig("plots/svg/methods_compare/fashion_censet_compare.svg")


# tikzplotlib.save("plots/tex/cen_set_compare/fashion_censet_compare.tex")
