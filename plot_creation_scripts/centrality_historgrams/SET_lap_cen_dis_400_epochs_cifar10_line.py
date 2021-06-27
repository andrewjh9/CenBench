import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import scipy.stats as stats


axis_label_size = 18
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : axis_label_size}

plt.rc('font', **font)


read_dataset_0_cifar10 = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_0__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_25_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_25__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_50_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_50__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_75_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_75__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_100_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_100__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_125_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_125__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_150_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_150__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_175_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_175__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_200_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_200__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_225_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_225__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_250_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_250__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_275_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_275__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_300_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_300__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_325_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_325__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_350_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_350__getting_distribution_cifar10_set.csv',delimiter='')
read_dataset_375_cifar10  = np.genfromtxt('results/base_line_set/cifar10/SET__cifar10_for_400_epochs_20210604-173718_num_sd_None_cen_dis_lap_epoch_375__getting_distribution_cifar10_set.csv',delimiter='')


max = max(max(read_dataset_0_cifar10),max(read_dataset_25_cifar10),max(read_dataset_50_cifar10),max(read_dataset_75_cifar10),max(read_dataset_100_cifar10),max(read_dataset_125_cifar10),max(read_dataset_150_cifar10),max(read_dataset_175_cifar10),max(read_dataset_200_cifar10),max(read_dataset_225_cifar10),max(read_dataset_250_cifar10),max(read_dataset_275_cifar10),max(read_dataset_300_cifar10),max(read_dataset_325_cifar10),max(read_dataset_350_cifar10),max(read_dataset_375_cifar10),)
max = int(max)

min = min(min(read_dataset_0_cifar10),min(read_dataset_25_cifar10),min(read_dataset_50_cifar10),min(read_dataset_75_cifar10),min(read_dataset_100_cifar10),min(read_dataset_125_cifar10),min(read_dataset_150_cifar10),min(read_dataset_175_cifar10),min(read_dataset_200_cifar10),min(read_dataset_225_cifar10),min(read_dataset_250_cifar10),min(read_dataset_275_cifar10),min(read_dataset_300_cifar10),min(read_dataset_325_cifar10),min(read_dataset_350_cifar10),min(read_dataset_375_cifar10),)

min = -20

internal = 0.5
plt.xlim(-1, 1)
plt.ylim(0, 10)
density = stats.gaussian_kde(read_dataset_0_cifar10)
plt.plot(read_dataset_0_cifar10, density(read_dataset_0_cifar10), label="0")

density = stats.gaussian_kde(read_dataset_25_cifar10)
plt.plot(read_dataset_25_cifar10, density(read_dataset_25_cifar10), label="25")

density = stats.gaussian_kde(read_dataset_50_cifar10)
plt.plot(read_dataset_50_cifar10, density(read_dataset_50_cifar10), label="50")

density = stats.gaussian_kde(read_dataset_75_cifar10)
plt.plot(read_dataset_75_cifar10, density(read_dataset_75_cifar10), label="75")

density = stats.gaussian_kde(read_dataset_100_cifar10)
plt.plot(read_dataset_100_cifar10, density(read_dataset_100_cifar10), label="100")

density = stats.gaussian_kde(read_dataset_125_cifar10)
plt.plot(read_dataset_125_cifar10, density(read_dataset_125_cifar10), label="125")

density = stats.gaussian_kde(read_dataset_150_cifar10)
plt.plot(read_dataset_150_cifar10, density(read_dataset_150_cifar10), label="150")

density = stats.gaussian_kde(read_dataset_175_cifar10)
plt.plot(read_dataset_175_cifar10, density(read_dataset_175_cifar10), label="175")

density = stats.gaussian_kde(read_dataset_200_cifar10)
plt.plot(read_dataset_75_cifar10, density(read_dataset_200_cifar10), label="200")

density = stats.gaussian_kde(read_dataset_225_cifar10)
plt.plot(read_dataset_100_cifar10, density(read_dataset_225_cifar10), label="225")

density = stats.gaussian_kde(read_dataset_250_cifar10)
plt.plot(read_dataset_125_cifar10, density(read_dataset_250_cifar10), label="250")

density = stats.gaussian_kde(read_dataset_275_cifar10)
plt.plot(read_dataset_150_cifar10, density(read_dataset_275_cifar10), label="275")

density = stats.gaussian_kde(read_dataset_300_cifar10)
plt.plot(read_dataset_175_cifar10, density(read_dataset_300_cifar10), label="300")

density = stats.gaussian_kde(read_dataset_325_cifar10)
plt.plot(read_dataset_175_cifar10, density(read_dataset_325_cifar10), label="325")

density = stats.gaussian_kde(read_dataset_350_cifar10)
plt.plot(read_dataset_175_cifar10, density(read_dataset_350_cifar10), label="350")

density = stats.gaussian_kde(read_dataset_375_cifar10)
plt.plot(read_dataset_175_cifar10, density(read_dataset_375_cifar10), label="375")


plt.xlabel("Laplacian centrality", fontsize=axis_label_size-2)
plt.ylabel("Probability density", fontsize=axis_label_size-2)
# plt.title("Frequency Distribution of Laplacian Centrality of Nodes in MLP on FashionMNIST at Epoch 175")
# plt.tight_layout()
# plt.ylim(None, 2000)
plt.legend(title="Epoch[#]", ncol=2, prop={'size': 12})
plt.grid()
# plt.show()

plt.savefig("plots/svg/histogram_lap/set_historgram_cfiar10_line.svg")
