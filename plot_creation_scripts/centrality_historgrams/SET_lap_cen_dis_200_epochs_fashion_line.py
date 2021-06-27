import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import scipy.stats as stats

axis_label_size = 18
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size}
plt.rc('font', **font)
read_dataset_0_fminst = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_0__sd_dis_.csv',delimiter='')
read_dataset_25_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_25__sd_dis_.csv',delimiter='')
read_dataset_50_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_50__sd_dis_.csv',delimiter='')
read_dataset_75_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_75__sd_dis_.csv',delimiter='')
read_dataset_100_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_100__sd_dis_.csv',delimiter='')
read_dataset_125_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_125__sd_dis_.csv',delimiter='')
read_dataset_150_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_150__sd_dis_.csv',delimiter='')
read_dataset_175_fminst  = np.genfromtxt('results/base_line_set/fashion/SET__fashion_mnist_for_200_epochs_20210603-164315_num_sd_None_cen_dis_lap_epoch_175__sd_dis_.csv',delimiter='')
    

min = min(min(read_dataset_0_fminst),min(read_dataset_25_fminst),min(read_dataset_50_fminst),min(read_dataset_75_fminst),min(read_dataset_100_fminst),min(read_dataset_125_fminst),min(read_dataset_150_fminst),min(read_dataset_175_fminst))
max = max(max(read_dataset_0_fminst),max(read_dataset_25_fminst),max(read_dataset_50_fminst),max(read_dataset_75_fminst),max(read_dataset_100_fminst),max(read_dataset_125_fminst),max(read_dataset_150_fminst),max(read_dataset_175_fminst))

internal = 0.5
plt.xlim(-10, 10)
plt.ylim(0, 1.2)
density = stats.gaussian_kde(read_dataset_0_fminst)
plt.plot(read_dataset_0_fminst, density(read_dataset_0_fminst), label="0")

density = stats.gaussian_kde(read_dataset_25_fminst)
plt.plot(read_dataset_25_fminst, density(read_dataset_25_fminst), label="25")

density = stats.gaussian_kde(read_dataset_50_fminst)
plt.plot(read_dataset_50_fminst, density(read_dataset_50_fminst), label="50")

density = stats.gaussian_kde(read_dataset_75_fminst)
plt.plot(read_dataset_75_fminst, density(read_dataset_75_fminst), label="75")

density = stats.gaussian_kde(read_dataset_100_fminst)
plt.plot(read_dataset_100_fminst, density(read_dataset_100_fminst), label="100")

density = stats.gaussian_kde(read_dataset_125_fminst)
plt.plot(read_dataset_125_fminst, density(read_dataset_125_fminst), label="125")

density = stats.gaussian_kde(read_dataset_150_fminst)
plt.plot(read_dataset_150_fminst, density(read_dataset_150_fminst), label="150")

density = stats.gaussian_kde(read_dataset_175_fminst)
plt.plot(read_dataset_175_fminst, density(read_dataset_175_fminst), label="175")

plt.xlabel("Laplacian centrality", fontsize=axis_label_size-2)
plt.ylabel("Probability density", fontsize=axis_label_size-2)
plt.legend(title="Epochs[#]",loc='upper right')
plt.grid()
# plt.title("Frequency Distribution of Laplacian Centrality of Nodes in MLP on FashionMNIST at Epoch 175")
# plt.tight_layout()

# plt.show()



# plt.title("Frequency Distribution of Laplacian Centrality of Nodes in SET on FashionMNIST at Epoch 175")
# plt.show()
# plt.savefig("plots/tex/histogram_lap/SET_historgram_fashionMNIST.svg")
plt.savefig("plots/svg/histogram_lap/SET_historgram_fashionMNIST_line.svg")
