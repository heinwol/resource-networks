from networkx.readwrite.gml import escape
import numpy as np
import pandas as pd
import sympy as sp
import networkx as nx
import pydot
import multiprocessing
from toolz import *
from IPython.display import Image, SVG
from typing import Sequence, Callable, Tuple, List, Union, Iterable, Any, Hashable
from dataclasses import dataclass

def parallelize_range(n_pools, rng):
    rng = list(rng)
    total_len = len(rng)
    size_of_pool = total_len // n_pools + int(bool(total_len % n_pools))
    return partition_all(size_of_pool, rng)

def const_iter(x):
    while True:
        yield x

Node = Hashable
FlowMatrix = Sequence[Sequence[Sequence[float]]]

class SimpleNodeArrayDescriptor:
    def __init__(self,
                 val_descriptor: dict[Hashable, int],
                 arr: np.ndarray | list[Any],
                 dims_affected: tuple[int] | None = None) -> None:
        self.val_descriptor = val_descriptor
        self.arr = arr
        self.dims_affected = set(dims_affected if dims_affected is not None else range(len(arr.shape)))
    def __getitem__(self, key: int | tuple[int]):
        # [0, 2] => (arr['lala', 5, 1] => arr[desc['lala'], 5, desc[12]])
        key_ = (key,) if not isinstance(key, tuple) else key
        new_key = list(key_)
        for i in range(len(key_)):
            if i in self.dims_affected:
                new_key[i] = self.val_descriptor[key_[i]]
        new_key = tuple(new_key)
        return self.arr[new_key if len(new_key) != 1 else new_key[0]]


@dataclass
class StateArray:

    node_descriptor: dict[Node, int]
    states_arr: np.ndarray
    flow_arr: np.ndarray
    total_output_res: list[float]

    def __len__(self) -> int:
        return len(self.states_arr)

    def __getitem__(self, time: int):
        return {'states': SimpleNodeArrayDescriptor(self.node_descriptor, self.states_arr[time], (0,)),
                'flow': SimpleNodeArrayDescriptor(self.node_descriptor, self.flow_arr[time], (0,1)),
                'total_output_res': SimpleNodeArrayDescriptor(self.node_descriptor, self.total_output_res, (0,))}


def parallel_plot(G: nx.DiGraph, states: StateArray, rng: list[int]):
    def my_fmt(x: float | int) -> str:
        if isinstance(x, int):
            return str(x)
        x_int, x_frac = int(x), x % 1
        if x_frac < 1e-3 or x >= 1e3:
            rem = ''
        else:
            len_int = len(str(x_int))
            rem = str(int(x_frac * 10**(4 - len_int)))
            rem = ('0'*(4 - len_int - len(rem)) + rem).rstrip('0')
        return str(x_int) + '.' + rem
    
    total_sum = states.states_arr[0].sum()
    
    def gen_width(amount_of_resource: float) -> float:
        # width in [0.35, 1.1]
        # assuming a linear function
        # 0 -> 0.35; total_sum -> 1
        return (1.1 - 0.35)/total_sum * amount_of_resource + 0.35
    
    res = [None]*len(rng)
    
    n_it = 0
    for idx in rng:
        state = states[idx]
        for v in G.nodes:
            G.nodes[v]['label'] = my_fmt(state[v])
            G.nodes[v]['width'] = gen_width(state[v])
            # G.nodes[v]['fillcolor'] = '#f0fff4' if 
            # 'fillcolor': '#f0fff4',
        for u, v, d in G.edges(data=True):
            d['label'] = d['weight']
        res[n_it] = SVG(nx.nx_pydot.to_pydot(G).create_svg())            
        n_it += 1
    return res



class ResourceDiGraph:
    
    def __init__(self, G: nx.DiGraph):
        # G = nx.from_numpy_matrix(np.array([[0, 2], [1, 0]]), create_using=nx.DiGraph)
        self.G: nx.DiGraph = G
        self.node_descriptor: dict[Node, int] = {node: i for i, node in enumerate(G.nodes)}
        self.idx_descriptor: list[Node] = [None]*len(G.nodes)
        for node, i in self.node_descriptor.items():
            self.idx_descriptor[i] = node
        
        for u, v, d in G.edges(data=True):
            if 'weight' not in d:
                d['weight'] = np.random.randint(1,10)
    
    def run_simulation(self, initial_state: dict[Node, float] | list[float], n_iters=30)\
        -> StateArray:
        if len(initial_state) != len(self.G.nodes):
            raise ValueError(
                'Incorrect initial states: expected states for ' +
                str(self.G.nodes) +
                ', while got:' +
                str(initial_state))
        n = len(initial_state)
        state_arr = np.zeros((n_iters, n))
        flow_arr = np.zeros((n_iters, n, n))

        if isinstance(initial_state, dict):
            state_dict = initial_state
        else:
            state_dict = {node: x for node, x in zip(self.node_descriptor.keys(), initial_state)}
        for j in range(n):
            state_arr[0, j] = state_dict[self.idx_descriptor[j]]
        
        total_output_res: list[float] = [sum(map(lambda v: self.G[u][v]['weight'], self.G[u]))
                               for u in self.idx_descriptor]
        print(total_output_res)
        for i in range(1, n_iters):
            for u in self.G.nodes:
                u_i = self.node_descriptor[u]
                for v in self.G[u]:
                    v_i = self.node_descriptor[v]
                    transferred_res = min(
                        self.G[u][v]['weight']/total_output_res[u_i] * state_arr[i-1, u_i],
                        self.G[u][v]['weight'])
                    flow_arr[i, u_i, v_i] = transferred_res
                    state_arr[i, v_i] += transferred_res
                state_arr[i, u_i] += max(state_arr[i-1, u_i] - total_output_res[u_i], 0)
                
        return StateArray(self.node_descriptor, state_arr, flow_arr, total_output_res)
    
    def plot_with_states(self, states: StateArray) -> Sequence[SVG]:
        G = self.G.copy()
        res = [None]*len(states)
        
#         G.graph['graph'] = {
#             'size': '900,1500!',
#             'dpi': 100
#         }
        G.graph['node'] = {
            'fontsize': 10,
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': '#f0fff4',
            'fixedsize': True
        }
        
        for v in G.nodes:
            G.nodes[v]['tooltip'] = str(v)
        for u, v in G.edges:
            G.edges[u, v]['label'] = self.G.edges[u, v]['weight']

        n_pools = min(8, len(states.states_arr))
        pool_obj = multiprocessing.Pool(n_pools)
        answer = pool_obj.starmap(
            parallel_plot,
            zip(const_iter(G),
                const_iter(states), 
                parallelize_range(n_pools, range(len(res)))))
        return np.concatenate(answer)
    

    def plot(self):
        G = self.G.copy()
        for u, v in G.edges:
            G.edges[u, v]['label'] = self.G.edges[u, v]['weight']
        return SVG(nx.nx_pydot.to_pydot(G).create_svg())
