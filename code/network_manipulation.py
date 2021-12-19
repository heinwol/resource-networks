from typing import Tuple
from toolz import *
import networkx as nx
from collections import deque
import numpy as np
import array_to_latex
import IPython
import pickle

def draw_weighted(G: nx.DiGraph, prop='weight'):
    pos = nx.spring_layout(G, iterations=150)
    if prop:
        edge_labels={(u, v): (d[prop] if prop in d else '')  
                        for u, v, d in G.edges(data=True)}
    else:
        edge_labels = {}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx(G, pos)

def make_random_weights(G: nx.DiGraph, gen = (lambda u, v: np.random.randint(0, 10))):
    G2 = G.copy()
    for u, v, d in G2.edges(data=True):
        d['weight'] = gen(u, v)
    return G2

def to_latex(a: np.ndarray):
    res = array_to_latex.to_ltx(a, arraytype='pmatrix', frmt='{:.4g}', print_out=False)
    return IPython.display.Latex(res)

# def pickle



# class GraphOp(nx.DiGraph):
#     def __init__(self, incoming_graph_data=None, **attr):
#         super().__init__(incoming_graph_data=incoming_graph_data, **attr)
    
#     def disjoint_union(self, rhs_: 'GraphOp') -> 'GraphOp':
#         lhs, rhs = self.copy(), rhs_.copy()

#         def get_node_idx(node):
#             if isinstance(node, tuple) and isinstance(node[1], int):
#                 return node[1]
#             else:
#                 return -1
    
#         def graph_to_indexed_form(G):
#             idxs = set(map(get_node_idx, G.nodes))




#         # def new_graph_node_idxs(graph: 'GraphOp') -> int:
#         #     if all(map(lambda x: isinstance(x, tuple) and isinstance(x[1], int), graph.nodes)):
#         #         return max(map(lambda x: x[1], graph.nodes)) + 1
#         #     else:
#         #         return 0
#         # idx1, idx2 = map(new_graph_node_idxs, [self, rhs])
#         # if idx1 == idx2:
#         #     idx2 += 1
#         # def new_node_idx(node, idx):
#         #     if idx == 0:
#         #         return (node, 0)
#         #     else:
#         #         return (node[0], idx)
#         # res = GraphOp()

#         for u, v, d in self.edges(data=True):


#     def __or__(self, rhs: 'GraphOp'):
#         return self.disjoint_union(rhs)