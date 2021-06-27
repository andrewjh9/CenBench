import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

axis_label_size = 24
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-6}
plt.rc('font', **font)

read_dataset_cen_sd_mlp = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_sd_lap__base_line.csv',delimiter='')



read_dataset_cen_sd_set= np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_sd_lap__sd_dis_.csv',delimiter='')


plt.plot(read_dataset_cen_sd_mlp,'k', label ="MLP")
plt.plot(read_dataset_cen_sd_set,'r', label ="SET", )
plt.legend(title="Method")
plt.grid()
plt.xlabel("Epoch[#]", fontsize=axis_label_size)
plt.ylabel("$\sigma$ ", fontsize=axis_label_size)
# plt.title("\sigma of Laplacian Centrality SET on Cifar10")
# plt.show()
plt.savefig("plots/svg/sd/sd_lap_cen_change_fashion.svg")
# tikzplotlib.save("plots/tex/histogram_lap/mlp_historgram_fashionMNIST_epoch_175.tex")
# tikzplotlib.save("plots/tex/sd/set_cifar_sd.tex")
