import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_mu = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_mean_lap__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_sd = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_sd_lap__getting_distribution_cifar10_set.csv',delimiter='')

for i in range(0, 200):
    if i % 25 == 0 : 
        print("epoch: "+str(i))
        print("$"+str(read_dataset_mu[i])+ "\pm "+str(read_dataset_sd[i])+"$")




print("final: ")
print("$"+str(np.mean(read_dataset_mu[-10:]))+ "\pm "+str(np.mean(read_dataset_sd[-10:]))+"$")


read_data_acc = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_accuracy__getting_distribution_cifar10_set.csv',delimiter='')

print(np.mean(read_data_acc[-10:]))

read_data_conn = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_connections__getting_distribution_cifar10_set.csv',delimiter='')

print(np.mean(read_data_conn))
