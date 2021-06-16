import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_cen_sd = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_sd_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')

plt.plot(read_dataset_cen_sd)
plt.xlabel("Epoch[\#]")
plt.ylabel("\sigma of Laplacian Centrality")
plt.title("\sigma of Laplacian Centrality MLP on cifar10")
plt.show()

# tikzplotlib.save("plots/tex/sd/mlp_cifar10_sd.tex")
