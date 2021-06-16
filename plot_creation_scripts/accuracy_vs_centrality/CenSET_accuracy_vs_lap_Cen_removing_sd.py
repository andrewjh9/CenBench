import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_CenSET_acc_epoch_300_removal_sd_2p5 = np.genfromtxt('results/censet_not_readding_connections/epochs300/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-212937_num_sd_2.5_accuracy_Not_readding_connections.csv',delimiter='')
read_dataset_CenSET_cen_epoch_300_removal_sd_2p5 = np.genfromtxt('results/censet_not_readding_connections/epochs300/CenSET_laplacian_fashion_mnist_for_300_epochs_20210602-212937_num_sd_2.5_lap_Not_readding_connections.csv',delimiter='')

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_CenSET_acc_epoch_300_removal_sd_2p5)
ax1.set_ylabel('Accuracy')


ax2 = ax1.twinx()
ax2.plot(read_dataset_CenSET_cen_epoch_300_removal_sd_2p5, 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r')



plt.xlabel("Epoch")
plt.title("CenSET Accuracy vs Laplacian Centrality (RM Nodes < Mean - sd *2.5) Not Readding connections")
plt.show()
# tikzplotlib.save("plots/tex/zeta_vs_accuracy_100_epochs.tex")
