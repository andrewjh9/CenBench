import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_lap = np.genfromtxt('results/SET_cifar10_for_300_epochs_laplacian_20210526-233159.csv',delimiter='')
read_dataset_acc = np.genfromtxt('results/SET_cifar10_for_300_epochs_20210526-233159_accuracy.csv',delimiter='')
read_dataset_conn = np.genfromtxt('results/SET_cifar10_for_300_epochs_20210526-233159_connections.csv',delimiter='')




# plt.plot(read_dataset_lap, label="Laplacian centrality")
# plt.plot(read_dataset_acc, label="Accuracy")
# plt.plot(read_dataset_conn, label="# Connections")
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_acc)
ax1.set_ylabel('Accuracy')

# ax1 = fig.add_subplot(111)
# ax1.plot(read_dataset_conn)
# ax1.set_ylabel('# Connections')

ax2 = ax1.twinx()
ax2.plot(read_dataset_lap, 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r')

plt.title("Accuracy vs Laplacian Centrality on Cifar10")
plt.xlabel("Epoch")
plt.show()
# tikzplotlib.save("plots/tex/lap_vs_accuracy_300_epochs_cifar.tex")
