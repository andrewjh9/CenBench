import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

read_dataset_acc_mlp = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_accuracy__base_line.csv',delimiter='')

read_dataset_acc_censet = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-171127_num_sd_2.7000000000000006_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')

read_dataset_acc_set = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_accuracy__sd_dis_.csv',delimiter='')

plt.plot(read_dataset_acc_mlp , label ="MLP")
plt.plot(read_dataset_acc_censet , label ="CenSET")
# plt.plot(read_dataset_accs_accset , label ="AccSET")
plt.plot(read_dataset_acc_set , label ="SET")
plt.legend()
plt.xlabel("Epoch[\#]")
plt.ylabel("Accuracy [\%]")
# plt.title("Different approaches on FashionMNIST")
# plt.show()
plt.savefig("plots/svg/methods_compare/fashion_censet_compare.svg")


# tikzplotlib.save("plots/tex/cen_set_compare/fashion_censet_compare.tex")
