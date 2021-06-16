import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

read_dataset_cen_sd = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_sd_lap__base_line.csv',delimiter='')

plt.plot(read_dataset_cen_sd)
plt.xlabel("Epoch[\#]")
plt.ylabel("\sigma of Laplacian Centrality")
plt.title("\sigma of Laplacian Centrality MLP on FashionMNIST")
# plt.show()

tikzplotlib.save("plots/tex/sd/mlp_fashion_sd.tex")
