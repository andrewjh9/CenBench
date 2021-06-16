import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_mu = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_mean_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_sd = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_sd_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')

for i in range(0, 400):
    if i % 25 == 0 : 
        print("epoch: "+str(i))
        print("$"+str(read_dataset_mu[i])+ "\pm "+str(read_dataset_sd[i])+"$")



print("final: ")
print("$"+str(read_dataset_mu[len(read_dataset_mu)-1])+ "\pm "+str(read_dataset_sd[len(read_dataset_sd)-1])+"$")

