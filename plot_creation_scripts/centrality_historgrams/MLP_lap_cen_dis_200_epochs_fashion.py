import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



read_dataset_0_fminst = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_0__base_line.csv',delimiter='')
read_dataset_25_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_25__base_line.csv',delimiter='')
read_dataset_50_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_50__base_line.csv',delimiter='')
read_dataset_75_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_75__base_line.csv',delimiter='')
read_dataset_100_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_100__base_line.csv',delimiter='')
read_dataset_125_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_125__base_line.csv',delimiter='')
read_dataset_150_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_150__base_line.csv',delimiter='')
read_dataset_175_fminst  = np.genfromtxt('results/base_line_MLP/fashion/MLP__fashion_mnist_for_200_epochs_20210604-093532_num_sd_None_cen_dis_lap_epoch_175__base_line.csv',delimiter='')

# read_dataset_set = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-190343_zeta_0.0.csv',delimiter='')
# 30
min = min(min(read_dataset_0_fminst),min(read_dataset_25_fminst),min(read_dataset_50_fminst),min(read_dataset_75_fminst),min(read_dataset_100_fminst),min(read_dataset_125_fminst),min(read_dataset_150_fminst),min(read_dataset_175_fminst))
max = max(max(read_dataset_0_fminst),max(read_dataset_25_fminst),max(read_dataset_50_fminst),max(read_dataset_75_fminst),max(read_dataset_100_fminst),max(read_dataset_125_fminst),max(read_dataset_150_fminst),max(read_dataset_175_fminst))

min = int(min)
max = int(max)
plt.hist(read_dataset_175_fminst , bins= np.arange(min, max, 0.5), label="175")
plt.hist(read_dataset_150_fminst , bins= np.arange(min, max, 0.5), label="150")
plt.hist(read_dataset_125_fminst , bins= np.arange(min, max, 0.5), label="125")
plt.hist(read_dataset_100_fminst , bins= np.arange(min, max, 0.5), label="100")
plt.hist(read_dataset_75_fminst , bins= np.arange(min, max, 0.5), label="75")
plt.hist(read_dataset_50_fminst , bins= np.arange(min, max, 0.5), label="50")
plt.hist(read_dataset_25_fminst , bins= np.arange(min, max, 0.5), label="25")
plt.hist(read_dataset_0_fminst , bins= np.arange(min, max, 0.5), label="0")

plt.legend( title="At Epoch[#]")

plt.xlabel("Laplacian centrality")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Laplacian Centrality of Nodes in MLP on FashionMNIST")
# plt.show()

tikzplotlib.save("plots/tex/histogram_lap/mlp_historgram_fashionMNIST.tex")
