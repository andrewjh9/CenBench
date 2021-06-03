import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


read_dataset_Set_sd = np.genfromtxt('results/set_epochs_200_recording_dis_sd/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_sd_lap__sd_dis_.csv',delimiter='')
perc_change_sd = np.diff(read_dataset_Set_sd) / read_dataset_Set_sd[:-1] * 100


plt.plot(perc_change_sd)
# plt.legend()
plt.ylabel("$\sigma$ change")
plt.xlabel("Epoch[#]")
plt.title("$\sigma$ change between epochs")
plt.show()
