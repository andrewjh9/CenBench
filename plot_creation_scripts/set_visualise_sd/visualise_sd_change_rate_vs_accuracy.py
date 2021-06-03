import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_Set_sd = np.genfromtxt('results/set_epochs_200_recording_dis_sd/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_sd_lap__sd_dis_.csv',delimiter='')
perc_change_sd = np.diff(read_dataset_Set_sd) / read_dataset_Set_sd[:-1] * 100

read_dataset_Set_acc = np.genfromtxt('results/set_epochs_200_recording_dis_sd/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_accuracy__sd_dis_.csv',delimiter='')


fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(perc_change_sd)
ax1.set_ylabel('$ \sigma $ Change')

# ax1 = fig.add_subplot(111)
# ax1.plot(read_dataset_conn)
# ax1.set_ylabel('# Connections')

ax2 = ax1.twinx()
ax2.plot(read_dataset_Set_acc, 'r-')
ax2.set_ylabel("Accuracy", color='r')
ax2.set_xlabel("Epoch[#]")
plt.legend()

plt.title("Accuracy vs $ \sigma $ Change")
plt.show()
