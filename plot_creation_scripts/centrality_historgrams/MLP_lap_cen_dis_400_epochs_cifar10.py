import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_0_cifar10 = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_0__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_25_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_25__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_50_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_50__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_75_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_75__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_100_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_100__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_125_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_125__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_150_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_150__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_175_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_175__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_200_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_200__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_225_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_225__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_250_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_250__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_275_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_275__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_300_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_300__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_325_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_325__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_350_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_350__mlp_saving_dis_at_multi_25.csv',delimiter='')
read_dataset_375_cifar10  = np.genfromtxt('results/base_line_MLP/cifar10/MLP__cifar10_for_400_epochs_20210604-191037_num_sd_None_cen_dis_lap_epoch_375__mlp_saving_dis_at_multi_25.csv',delimiter='')

# read_dataset_set = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-190343_zeta_0.0.csv',delimiter='')
# 30

max = max(max(read_dataset_0_cifar10),max(read_dataset_25_cifar10),max(read_dataset_50_cifar10),max(read_dataset_75_cifar10),max(read_dataset_100_cifar10),max(read_dataset_125_cifar10),max(read_dataset_150_cifar10),max(read_dataset_175_cifar10),max(read_dataset_200_cifar10),max(read_dataset_225_cifar10),max(read_dataset_250_cifar10),max(read_dataset_275_cifar10),max(read_dataset_300_cifar10),max(read_dataset_325_cifar10),max(read_dataset_350_cifar10),max(read_dataset_375_cifar10),)
max = int(max)

min = min(min(read_dataset_0_cifar10),min(read_dataset_25_cifar10),min(read_dataset_50_cifar10),min(read_dataset_75_cifar10),min(read_dataset_100_cifar10),min(read_dataset_125_cifar10),min(read_dataset_150_cifar10),min(read_dataset_175_cifar10),min(read_dataset_200_cifar10),min(read_dataset_225_cifar10),min(read_dataset_250_cifar10),min(read_dataset_275_cifar10),min(read_dataset_300_cifar10),min(read_dataset_325_cifar10),min(read_dataset_350_cifar10),min(read_dataset_375_cifar10),)



min = int(min)


plt.hist(read_dataset_375_cifar10 , bins= np.arange(min, max, 0.5), label="375")
plt.hist(read_dataset_350_cifar10 , bins= np.arange(min, max, 0.5), label="350")
plt.hist(read_dataset_325_cifar10 , bins= np.arange(min, max, 0.5), label="325")
plt.hist(read_dataset_300_cifar10 , bins= np.arange(min, max, 0.5), label="300")
plt.hist(read_dataset_275_cifar10 , bins= np.arange(min, max, 0.5), label="275")
plt.hist(read_dataset_250_cifar10 , bins= np.arange(min, max, 0.5), label="250")
plt.hist(read_dataset_225_cifar10 , bins= np.arange(min, max, 0.5), label="225")
plt.hist(read_dataset_200_cifar10 , bins= np.arange(min, max, 0.5), label="200")
plt.hist(read_dataset_175_cifar10 , bins= np.arange(min, max, 0.5), label="175")
plt.hist(read_dataset_150_cifar10 , bins= np.arange(min, max, 0.5), label="150")
plt.hist(read_dataset_125_cifar10 , bins= np.arange(min, max, 0.5), label="125")
plt.hist(read_dataset_100_cifar10 , bins= np.arange(min, max, 0.5), label="100")
plt.hist(read_dataset_75_cifar10 , bins= np.arange(min, max, 0.5), label="75")
plt.hist(read_dataset_50_cifar10 , bins= np.arange(min, max, 0.5), label="50")
plt.hist(read_dataset_25_cifar10 , bins= np.arange(min, max, 0.5), label="25")
plt.hist(read_dataset_0_cifar10 , bins= np.arange(min, max, 0.5), label="0")

plt.legend( title="At Epoch[#]")

plt.xlabel("Laplacian centrality")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Laplacian Centrality of Nodes in MLP on FashionMNIST")
# plt.show()

tikzplotlib.save("plots/tex/histogram_lap/mlp_historgram_cifar10.tex")
