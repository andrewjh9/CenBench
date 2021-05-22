import csv
import matplotlib.pyplot as plt
import numpy as np


for i in range(0,10):
    with open(i+'laplacian.csv', newline=',') as f:
        reader = csv.reader(f)
        data = list(reader)