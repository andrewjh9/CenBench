import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_mu = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_mean_lap__base_line.csv',delimiter='')
read_dataset_sd = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_sd_lap__base_line.csv',delimiter='')

for i in range(0, 200):
    if i % 25 == 0 : 
        print("epoch: "+str(i))
        print("$"+str(read_dataset_mu[i])+ "\pm "+str(read_dataset_sd[i])+"$")



print("final: ")
print("$"+str(read_dataset_mu[len(read_dataset_mu)-1])+ "\pm "+str(read_dataset_sd[len(read_dataset_sd)-1])+"$")

read_data_acc = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_accuracy__base_line.csv',delimiter='')

print(read_data_acc[-1])

read_data_conn = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_connections__base_line.csv',delimiter='')

print(read_data_conn[-1])