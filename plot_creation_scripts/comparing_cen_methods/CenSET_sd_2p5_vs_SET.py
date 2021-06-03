import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_SET_cen_epoch_300 = np.genfromtxt('results/base_line_set/SET__fashion_mnist_for_1000_epochs_20210531-183228_zeta__lap.csv',delimiter='')[0:300]
read_dataset_CenSET_cen_epoch_300_removal_not_readd_sd_2p5 = np.genfromtxt('results/censet_not_readding_connections/epochs300/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-212937_num_sd_2.5_lap_Not_readding_connections.csv',delimiter='')
read_dataset_CenSET_cen_epoch_300_removal_sd_2p5 = np.genfromtxt('',delimiter='')

plt.plot(read_dataset_SET_cen_epoch_300, label="SET")
plt.plot(read_dataset_CenSET_cen_epoch_300_removal_not_readd_sd_2p5, label="CenSET prune = sd*2.5 Not readding")
plt.plot(read_dataset_CenSET_cen_epoch_300_removal_sd_2p5, label="CenSET prune = sd*2.5")

plt.legend()
plt.ylabel("Laplacian Centrality")
plt.xlabel("Epoch")
plt.title("Laplacian Centrality on Epochs SET vs CenSET (readding and not)")
plt.show()
# tikzplotlib.save("plots/tex/zeta_vs_accuracy_100_epochs.tex")
