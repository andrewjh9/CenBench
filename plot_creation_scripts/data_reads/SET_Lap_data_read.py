import csv
import matplotlib.pyplot as plt
import numpy as np

import collections



# frequency = collections.Counter(read_csv())
# print(read_csv()[0])
# # print(frequency)
# data0 = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/0laplacian.csv',delimiter='')
# data99 = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/99laplacian.csv',delimiter='')


# print("epoch_0_var: ", np.var(data0))
# print("epoch_0_std: ", np.std(data0))

# print("---------------------")

# print("epoch_99_var : ", np.var(data99))
# print("epoch_99_std : ", np.std(data99))


cen_means = []

# for i in range(0,99):
#     read_dataset = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/'+str(i)+'laplacian.csv',delimiter='')
#     # print(str(i)+"i: ", np.mean(read_dataset))
#     cen_means.append(np.mean(read_dataset))

# print("datasets.size")
# cen_means = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/laplacian_fashion_set.csv',delimiter='')

# plt.plot(cen_means)
# # plt.show()
# plt.ylabel("Laplasian Centrality")
# plt.xlabel("Epoch")

# plt.savefig("./plots/laplacian_fashion_set_300_epochs.png")
acc = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/SET_accuracy_fashion_mnist_for_300_epochs_20210525-185840_.csv',delimiter='')
cen_means = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/laplacian_fashion_set.csv',delimiter='')

# plt.plot(cen_means,"Laplacian Centrality")
# # plt.show()
# plt.plot(acc,"Accuracy")
# fig = plt.figure()
# fig = plt.figure()

# ax1 = fig.add_subplot(111)
# ax1.plot(acc[2:])
# ax1.set_ylabel('Accuracy')

# ax2 = ax1.twinx()
# ax2.plot(cen_means[2:], 'r-')
# ax2.set_ylabel("Laplacian Centrality", color='r')
# # plt.show()')

# plt.xlabel("Epoch")
plt.savefig("./plots/SET_accuracy_vs_lap_fashion_300_epochs.png")
# # plt.boxplot(data0)
# plt.boxplot(data99)  # density=False would make counts

# plt.show()
