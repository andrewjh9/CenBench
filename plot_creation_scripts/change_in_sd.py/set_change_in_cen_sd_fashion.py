import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_cen_sd = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_sd_lap__sd_dis_.csv',delimiter='')

plt.plot(read_dataset_cen_sd)
plt.xlabel("Epoch[\#]")
plt.ylabel("\sigma of Laplacian Centrality")
plt.title("\sigma of Laplacian Centrality SET fasionMNIST")
# plt.show()

# tikzplotlib.save("plots/tex/histogram_lap/mlp_historgram_fashionMNIST_epoch_175.tex")
tikzplotlib.save("plots/tex/sd/set_fashion_sd.tex")
