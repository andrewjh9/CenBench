import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_cen_sd = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_sd_lap__getting_distribution_cifar10_set.csv',delimiter='')

plt.plot(read_dataset_cen_sd)
plt.xlabel("Epoch[\#]")
plt.ylabel("\sigma of Laplacian Centrality")
plt.title("\sigma of Laplacian Centrality SET on Cifar10")
# plt.show()

# tikzplotlib.save("plots/tex/histogram_lap/mlp_historgram_fashionMNIST_epoch_175.tex")
tikzplotlib.save("plots/tex/sd/set_cifar_sd.tex")
