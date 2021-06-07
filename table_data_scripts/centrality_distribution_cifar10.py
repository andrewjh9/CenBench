import numpy as np


read_dataset_mlp_cifar_10_mean = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_mean_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_mlp_cifar_10_sd = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_sd_lap__mlp_saving_dis_at_multi_25.csv',delimiter='')


read_dataset_SET_cifar_10_mean = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_mean_lap__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_SET_cifar_10_sd = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_sd_lap__getting_distribution_cifar10_set.csv',delimiter='')


# read_dataset_AccSET_cifar_10_mean = np.genfromtxt('',delimiter='')
# read_dataset_AccSET_cifar_10_sd = np.genfromtxt('',delimiter='')


# read_dataset_CenSET_cifar_10_mean = np.genfromtxt('',delimiter='')
# read_dataset_CenSET_cifar_10_sd = np.genfromtxt('',delimiter='')


print("MLP cifar10 mean : sd ")
print("0 =  " + str(round(read_dataset_mlp_cifar_10_mean[0],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[0],3)) )
print("25 =  " + str(round(read_dataset_mlp_cifar_10_mean[25],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[25],3)) )
print("50 =  " + str(round(read_dataset_mlp_cifar_10_mean[50],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[50],3)) )
print("75 =  " + str(round(read_dataset_mlp_cifar_10_mean[75],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[75],3)) )
print("100 =  " + str(round(read_dataset_mlp_cifar_10_mean[100],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[100],3)) )
print("125 =  " + str(round(read_dataset_mlp_cifar_10_mean[125],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[125],3)) )
print("150 =  " + str(round(read_dataset_mlp_cifar_10_mean[150],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[150],3)) )
print("175 =  " + str(round(read_dataset_mlp_cifar_10_mean[175],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[175],3)) )
print("200 =  " + str(round(read_dataset_mlp_cifar_10_mean[200],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[200],3)) )
print("225 =  " + str(round(read_dataset_mlp_cifar_10_mean[225],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[225],3)) )
print("250 =  " + str(round(read_dataset_mlp_cifar_10_mean[250],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[250],3)) )
print("275 =  " + str(round(read_dataset_mlp_cifar_10_mean[275],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[275],3)) )
print("300 =  " + str(round(read_dataset_mlp_cifar_10_mean[300],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[300],3)) )
print("325 =  " + str(round(read_dataset_mlp_cifar_10_mean[325],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[325],3)) )
print("350 =  " + str(round(read_dataset_mlp_cifar_10_mean[350],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[350],3)) )
print("375 =  " + str(round(read_dataset_mlp_cifar_10_mean[375],3)) + " : " + str(round(read_dataset_mlp_cifar_10_sd[375],3)) )



print("SET cifar10 mean : sd ")
print("0 =  " + str(round(read_dataset_SET_cifar_10_mean[0],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[0],3) ))
print("25 =  " + str(round(read_dataset_SET_cifar_10_mean[25],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[25],3) ))
print("50 =  " + str(round(read_dataset_SET_cifar_10_mean[50],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[50],3) ))
print("75 =  " + str(round(read_dataset_SET_cifar_10_mean[75],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[75],3) ))
print("100 =  " + str(round(read_dataset_SET_cifar_10_mean[100],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[100],3) ))
print("125 =  " + str(round(read_dataset_SET_cifar_10_mean[125],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[125],3) ))
print("150 =  " + str(round(read_dataset_SET_cifar_10_mean[150],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[150],3) ))
print("175 =  " + str(round(read_dataset_SET_cifar_10_mean[175],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[175],3) ))
print("200 =  " + str(round(read_dataset_SET_cifar_10_mean[200],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[200],3)) )
print("225 =  " + str(round(read_dataset_SET_cifar_10_mean[225],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[225],3)) )
print("250 =  " + str(round(read_dataset_SET_cifar_10_mean[250],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[250],3)) )
print("275 =  " + str(round(read_dataset_SET_cifar_10_mean[275],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[275],3)) )
print("300 =  " + str(round(read_dataset_SET_cifar_10_mean[300],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[300],3)) )
print("325 =  " + str(round(read_dataset_SET_cifar_10_mean[325],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[325],3)) )
print("350 =  " + str(round(read_dataset_SET_cifar_10_mean[350],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[350],3)) )
print("375 =  " + str(round(read_dataset_SET_cifar_10_mean[375],3)) + " : " + str(round(read_dataset_SET_cifar_10_sd[375],3)) )