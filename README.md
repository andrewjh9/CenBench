# CenBench
CenBench is the code base for performing the experiments for my (Andrew Heath) bachelor thesis titled : "Investigating the use of hub neuron identification for pruning sparse neural networks" under the supervision of Dr. Elena. Mocanu at the University of Twente (Netherlands).  


This Github contains the Juypyter Notebooks used to gather data for the experiments of the thesis. a copy of thesis can be found here: [INSERT LINK ONCE UPLOAD TO UT SHOWCASE]. The code is made public for the sake of reproducibility. The results of the experiments conducted are also in this github. Please contact me at a.j.heath@student.utwente.nl if you would like more information. 

If you are looking for a starting point for your own research on Sparse Neural Networks, I would recommend the following repo [https://github.com/dcmocanu/centrality-metrics-complex-networks](https://github.com/dcmocanu/centrality-metrics-complex-networks) by Dr. Decebal Mocanu where you can find a number of efficient implementations of Sparse Neural Networks. The repo was also used as the a starting point for the creation of the Jupyter Notebooks in this repo.

## Structure of this repo 
This repo contains the following: 

- CenBench.ipynb, the Jupyter Notebook used for the experiment of my thesis. 
- CenBench_MLP.pynb, a variation of the CenBench.ipynb but for fully connected MLPs
- /results, contains the CSVs of all results from the experiments conducted.
- /plot_creation_scripts, contains python scripts used to create the plot as seen in the thesis
