import numpy as np


read_dataset_mlp_fashion_mean = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_mean_lap__base_line.csv',delimiter='')
read_dataset_mlp_fashion_sd = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_sd_lap__base_line.csv',delimiter='')


read_dataset_SET_fashion_mean = np.genfromtxt('results/set_epochs_200_recording_dis_sd/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_mean_lap__sd_dis_.csv',delimiter='')
read_dataset_SET_fashion_sd = np.genfromtxt('results/set_epochs_200_recording_dis_sd/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_sd_lap__sd_dis_.csv',delimiter='')


# read_dataset_AccSET_fashion_mean = np.genfromtxt('',delimiter='')
# read_dataset_AccSET_fashion_sd = np.genfromtxt('',delimiter='')


# read_dataset_CenSET_fashion_mean = np.genfromtxt('',delimiter='')
# read_dataset_CenSET_fashion_sd = np.genfromtxt('',delimiter='')



print("MLP fashion mean : sd ")
print("0 =  " + str(round(read_dataset_mlp_fashion_mean[0],3))+ " : " + str(round(read_dataset_mlp_fashion_sd[0],3)) )
print("25 =  " + str(round(read_dataset_mlp_fashion_mean[25],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[25],3)) )
print("50 =  " + str(round(read_dataset_mlp_fashion_mean[50],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[50],3)) )
print("75 =  " + str(round(read_dataset_mlp_fashion_mean[75],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[75],3)) )
print("100 =  " + str(round(read_dataset_mlp_fashion_mean[100],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[100],3)) )
print("125 =  " + str(round(read_dataset_mlp_fashion_mean[125],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[125],3)) )
print("150 =  " + str(round(read_dataset_mlp_fashion_mean[150],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[150],3)) )
print("175 =  " + str(round(read_dataset_mlp_fashion_mean[175],3)) + " : " + str(round(read_dataset_mlp_fashion_sd[175],3)) )

print("SET fashion mean : sd ")
print("0 =  " + str(round(read_dataset_SET_fashion_mean[0],3)) + " : " + str(round(read_dataset_SET_fashion_sd[0],3)) )
print("25 =  " + str(round(read_dataset_SET_fashion_mean[25],3)) + " : " + str(round(read_dataset_SET_fashion_sd[25],3)) )
print("50 =  " + str(round(read_dataset_SET_fashion_mean[50],3)) + " : " + str(round(read_dataset_SET_fashion_sd[50],3)) )
print("75 =  " + str(round(read_dataset_SET_fashion_mean[75],3)) + " : " + str(round(read_dataset_SET_fashion_sd[75],3)) )
print("100 =  " + str(round(read_dataset_SET_fashion_mean[100],3)) + " : " + str(round(read_dataset_SET_fashion_sd[100],3)) )
print("125 =  " + str(round(read_dataset_SET_fashion_mean[125],3)) + " : " + str(round(read_dataset_SET_fashion_sd[125],3)) )
print("150 =  " + str(round(read_dataset_SET_fashion_mean[150],3)) + " : " + str(round(read_dataset_SET_fashion_sd[150],3)) )
print("175 =  " + str(round(read_dataset_SET_fashion_mean[175],3)) + " : " + str(round(read_dataset_SET_fashion_sd[175],3) ))