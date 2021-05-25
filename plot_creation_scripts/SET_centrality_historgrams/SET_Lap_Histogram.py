import csv
import matplotlib.pyplot as plt
import numpy as np

import collections



# frequency = collections.Counter(read_csv())
# print(read_csv()[0])
# # print(frequency)
data0 = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/0laplacian.csv',delimiter='')
data99 = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/99laplacian.csv',delimiter='')

# plt.hist(data0, density=False, label='Epoch 0') 
plt.hist(data99, density=False, label='Epoch 99')  # density=False would make counts
plt.ylabel('Frequency')
plt.xlabel('Laplacian Centrality');
# plt.savefig("./plot_creation_scripts/plots/lap_epoch_0_vs)_99.png")
# plt.legend(loc='upper right')

plt.show()
