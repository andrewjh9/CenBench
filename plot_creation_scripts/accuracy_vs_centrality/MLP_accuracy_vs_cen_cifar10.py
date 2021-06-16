import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from itertools import groupby
import pandas as pd
axis_label_size = 22
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size-2}

plt.rc('font', **font)

read_dataset_cen =  np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_mean_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_cen = [k for k,g in groupby(read_dataset_cen) if k!=0] # Removing duplicated in dataset
read_dataset_cen_fixed = [None]*400
for i in range(400):
    value = None
    if  not(i % 25 ):
        value  = read_dataset_cen[i//25] 
    read_dataset_cen_fixed[i] = value
read_dataset_cen = read_dataset_cen_fixed

def splitSerToArr(ser):
    return [ser.index, ser.as_matrix()]

read_dataset_acc = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_accuracy__mlp_saving_dis_at_multi_25.csv',delimiter='')

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(read_dataset_acc*100, 'k-')
ax1.set_ylabel('Accuracy [%]', fontsize=axis_label_size)
ax1.set_xlabel("Epoch [#]", fontsize=axis_label_size)

xs = range(400)

s1 = pd.Series(read_dataset_cen, index=xs)

ax2 = ax1.twinx()
ax2.plot(*splitSerToArr(s1.dropna()), 'r-')
ax2.set_ylabel("Laplacian Centrality", color='r', fontsize=axis_label_size)
ax2.set_xlabel("Epoch [#]", fontsize=axis_label_size)



plt.xlabel("Epoch")
# plt.show()
plt.grid()

plt.savefig("plots/svg/cen_vs_acc/mlp_cifar10_acc_cen.svg")
# tikzplotlib.save("plots/svg/cen_vs_acc/mlp_cifar10_acc_cen.svg")
