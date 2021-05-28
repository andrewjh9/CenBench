import csv
import matplotlib.pyplot as plt
import numpy as np

import collections
import scipy


cen_means = np.genfromtxt('/media/andrew/Storage/MOD-12-TCS/Research Project/CenBench/results/node_centrality_scores_set/laplacian_fashion_set.csv',delimiter='')
x = range(50, 300)
y = cen_means[50:300]

# z = np.polyfit(x, y, 3)
# f = np.poly1d(z)

# # calculate new x's and y's
# x_new = np.linspace(x[0], x[-1], 50)
# y_new = f(x_new)

# plt.plot(x,y,'o', x_new, y_new)
# plt.xlim([x[0]-1, x[-1] + 1 ])
# plt.show()
# print(np.polyfit(x, y, 3, rcond=None, full=False))
