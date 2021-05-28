import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib




read_dataset_0_cifar = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/cifar10_0laplacian_20210526-171315.csv',delimiter='')
read_dataset_50_cifar  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/cifar10_50laplacian_20210526-181455.csv',delimiter='')
read_dataset_100_cifar  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/cifar10_100laplacian_20210526-191912.csv',delimiter='')
read_dataset_150_cifar  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/cifar10_150laplacian_20210526-202222.csv',delimiter='')
read_dataset_200_cifar  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/cifar10_200laplacian_20210526-212648.csv',delimiter='')
read_dataset_250_cifar  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/cifar10_250laplacian_20210526-222951.csv',delimiter='')


read_dataset_0_fminst = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/fashion_mnist_0laplacian_20210526-233239.csv',delimiter='')
read_dataset_50_fminst  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/fashion_mnist_50laplacian_20210527-001743.csv',delimiter='')
read_dataset_100_fminst  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/fashion_mnist_100laplacian_20210527-010234.csv',delimiter='')
read_dataset_150_fminst  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/fashion_mnist_150laplacian_20210527-014649.csv',delimiter='')
read_dataset_200_fminst  = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/fashion_mnist_200laplacian_20210527-023247.csv',delimiter='')
read_dataset_250_fminst = np.genfromtxt('results/node_centrality_scores_set/exp_26_may_SET/fashion_mnist_250laplacian_20210527-031755.csv',delimiter='')

# read_dataset_set = np.genfromtxt('results/zeta/CenSET_laplacian_accuracy_cifar10_for_100_epochs_20210522-190343_zeta_0.0.csv',delimiter='')
plt.hist(read_dataset_250_cifar, bins = 50)



plt.xlabel("Laplacian centrality")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Centrality of Nodes at Epoch 250")
# plt.show()

tikzplotlib.save("plots/tex/histogram_lap/cifar_250_epochs.tex")
