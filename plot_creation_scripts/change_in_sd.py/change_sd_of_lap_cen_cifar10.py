import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}
plt.rc('font', **font)

read_dataset_cen_sd_mlp = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_sd_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_cen_sd_set = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_sd_lap__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_cen_sd_mlp = [x for i, x in enumerate(read_dataset_cen_sd_mlp) if i % 25 == 0 ]
a_list = range(0, 16)
epochs = [element * 25 for element in a_list]


plt.plot(epochs,read_dataset_cen_sd_mlp, 'k', label ="MLP")
plt.plot(read_dataset_cen_sd_set,'r', label ="SET")
plt.legend(title="Method")
plt.xlabel("Epoch[#]", fontsize=axis_label_size)
plt.ylabel("$\sigma$", fontsize=axis_label_size)
# plt.title("\sigma of Laplacian Centrality SET on Cifar10")
# plt.show()
plt.grid()
plt.savefig("plots/svg/sd/sd_lap_cen_change_cifar10.svg")


# tikzplotlib.save("plots/tex/sd/mlp_cifar10_sd.tex")
