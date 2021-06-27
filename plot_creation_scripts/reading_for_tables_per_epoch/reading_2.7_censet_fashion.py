import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_mu = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-181713_num_sd_2.8000000000000007_mean_lap__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_sd = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-181713_num_sd_2.8000000000000007_sd_lap__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')

for i in range(0, 200):
    if i % 25 == 0 : 
        print("epoch: "+str(i))
        print("$"+str(read_dataset_mu[i])+ " \pm "+str(read_dataset_sd[i])+"$")



print("final: ")
print("$"+str(read_dataset_mu[len(read_dataset_mu)-1])+ "\pm "+str(read_dataset_sd[len(read_dataset_sd)-1])+"$")

read_data_acc = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-160538_num_sd_2.6000000000000005_accuracy__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')

print(np.mean(read_data_acc[-10:]))

read_data_conn = np.genfromtxt('results/find_sd_prune_value/fashion/CenSET_laplacian_fashion_mnist_for_200_epochs_20210608-160538_num_sd_2.6000000000000005_connections__narrow_search_missing_valuses_finding_opti_sd_removal_rate.csv',delimiter='')

print(np.mean(read_data_conn[-10:]))