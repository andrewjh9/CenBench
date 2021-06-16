import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality import betweenness
G = nx.Graph()
G.add_nodes_from([1, 6])


G.add_edge(1,2)
G.add_edge(3,2)
G.add_edge(5,2)
G.add_edge(6,2)
G.add_edge(4,2)
G.add_edge(4,1)
G.add_edge(5,6)


# bet = nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
# print(bet)
# H = nx.relabel_nodes(G, bet)

nx.draw(G, with_labels=True, font_weight='bold')

plt.savefig("example_graph.png")