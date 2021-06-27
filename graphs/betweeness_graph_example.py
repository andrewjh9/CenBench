import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality import betweenness
from tensorflow.python.keras.utils.np_utils import normalize
import networkit as nk

G = nx.Graph()
G.add_nodes_from([1, 6])


G.add_edge(4,1, weight=0.4)
G.add_edge(4,2, weight=0.4)
G.add_edge(4,5, weight=0.2)
G.add_edge(4,6, weight=0.2)
G.add_edge(5,2, weight=0.2)
G.add_edge(3,6, weight=0.6)
G.add_edge(3,4, weight=0.6)

# bet = nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
# print(bet)
# H = nx.relabel_nodes(G, bet)
pos=nx.get_node_attributes(G,'pos')


# nx.betweenness_centrality(G,normalized=False)
print(nx.betweenness_centrality(G,normalized=False))
nkG = nk.nxadapter.nx2nk(G)
btwn = nk.centrality.LaplacianCentrality(nkG)
btwn.run()
ranking = btwn.ranking() 
print(ranking)
# plt.savefig("example_graph.png")