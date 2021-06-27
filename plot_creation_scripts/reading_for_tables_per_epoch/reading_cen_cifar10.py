import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

import math

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


read_dataset_mu = np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-024647_num_sd_3.2_mean_lap__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')
read_dataset_sd = np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-024647_num_sd_3.2_sd_lap__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

for i in range(0, 400):
    if i % 25 == 0 : 
        print("epoch: "+str(i))
        print(str(truncate(read_dataset_mu[i],2))+ "$\pm$"+str(truncate(read_dataset_sd[i],2)))



print("final: ")
print("$"+str(np.mean(read_dataset_mu[-10:]))+ "\pm "+str(np.mean(read_dataset_sd[-10:]))+"$")

read_data_acc = np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-024647_num_sd_3.2_accuracy__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

print(np.mean(read_data_acc[-10:]))


read_data_conn = np.genfromtxt('results/find_sd_prune_value/cifar10/narrow/CenSET_laplacian_cifar10_for_400_epochs_20210617-024647_num_sd_3.2_connections__narrow_search_finding_opti_sd_removal_rate.csv',delimiter='')

print(np.mean(read_data_conn))
