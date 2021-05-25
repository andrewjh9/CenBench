import csv
import matplotlib.pyplot as plt
import numpy as np

import collections



# frequency = collections.Counter(read_csv())
# print(read_csv()[0])
# # print(frequency)
# data0 = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/0kat.csv',delimiter='')
# data99 = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/99laplacian.csv',delimiter='')


# print("epoch_0_var: ", np.var(data0))
# print("epoch_0_std: ", np.std(data0))

# print("---------------------")

# print("epoch_99_var : ", np.var(data99))
# print("epoch_99_std : ", np.std(data99))


cen_means = []

for i in range(0,100):
    read_dataset = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/'+str(i)+'KatzCentrality.csv',delimiter='')
    # print(str(i)+"i: ", np.mean(read_dataset))
    cen_means.append(np.mean(read_dataset))

# print("datasets.size")

plt.plot(cen_means)
plt.show()

# # plt.boxplot(data0)
# plt.boxplot(data99)  # density=False would make counts

# plt.show()
