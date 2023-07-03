import numpy as np
from scipy.sparse.linalg import eigs as sparce_eigs
import pandas as pd
import sympy as sp
import networkx as nx
import pydot
import multiprocessing
import operator
import itertools
from toolz import *
from IPython.display import Image, SVG
from typing import Dict, Optional, Sequence, Callable, Tuple, List, Union, Iterable, Any, Hashable
from dataclasses import dataclass
from ipywidgets import interact, widgets
import pandas as pd

MAX_NODE_WIDTH = 1.1

lmap = compose(list, map)
tmap = compose(tuple, map)
get = curry(operator.getitem)

def linear_func_from_2_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> \
    Callable[[float], float]:
    if np.allclose(p2[0], p1[0]):
        if np.allclose(p2[1], p1[1]):
            return lambda x: p1[1]
        else:
            raise ValueError(f'Invalid points for linear function: {p1}, {p2}')
    else:
        k = (p2[1] - p1[1])/(p2[0] - p1[0])
        b = (p1[1]*p2[0] - p2[1]*p1[0])/(p2[0] - p1[0])
        return lambda x: k*x + b

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
                 val_descriptor: Dict[Hashable, int],
                 arr: Union[np.ndarray, List[Any]],
                 dims_affected: Optional[Tuple[int]] = None) -> None:
        self.val_descriptor = val_descriptor
        self.arr = arr
        self.dims_affected = set(dims_affected if dims_affected is not None else range(len(arr.shape)))
    def __getitem__(self, key: Union[int, Tuple[int]]):
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

    node_descriptor: Dict[Node, int]
    idx_descriptor: Dict[int, Node]
    states_arr: np.ndarray
    flow_arr: np.ndarray
    total_output_res: List[float]

    def __len__(self) -> int:
        return len(self.states_arr)

    def __getitem__(self, time: int):
        return {'states': SimpleNodeArrayDescriptor(self.node_descriptor, self.states_arr[time], (0,)),
                'flow': SimpleNodeArrayDescriptor(self.node_descriptor, self.flow_arr[time], (0,1)),
                'total_output_res': SimpleNodeArrayDescriptor(self.node_descriptor, self.total_output_res, (0,))}


def parallel_plot(G: nx.DiGraph, states: StateArray, rng: List[int]):
    def my_fmt(x: Union[float,int]) -> str:
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
    
    total_sum = states.states_arr[-1].sum()
    calc_node_width = linear_func_from_2_points((0, 0.35), (total_sum, 1.1))
    res = [None]*len(rng)
    n_it = 0
    for idx in rng:
        state = states[idx]
        for v in G.nodes:
            if 'color' not in G.nodes[v] or G.nodes[v]['color'] != 'transparent':
                G.nodes[v]['label'] = my_fmt(state['states'][v])
                G.nodes[v]['width'] = calc_node_width(state['states'][v])
                
                G.nodes[v]['fillcolor'] = ('#f0fff4' if 
                    state['states'][v] < state['total_output_res'][v] else '#b48ead')

        for u, v, d in G.edges(data=True):
            d['label'] = d['weight']
        res[n_it] = SVG(nx.nx_pydot.to_pydot(G).create_svg())            
        n_it += 1
    return res



class ResourceDiGraph:
    
    def __init__(self, G: Optional[nx.DiGraph] = None):
        if G is None:
            G = nx.DiGraph()
        self.G: nx.DiGraph = G
        self.node_descriptor: dict[Node, int] = {node: i for i, node in enumerate(G.nodes)}
        self.idx_descriptor: list[Node] = [None]*len(G.nodes)
        for node, i in self.node_descriptor.items():
            self.idx_descriptor[i] = node
        
        for u, v, d in G.edges(data=True):
            if 'weight' not in d:
                d['weight'] = np.random.randint(1,10)
        self.stochastic_matrix = None
        self.adjacency_matrix = None
        self.recalculate_matrices()
    
    def recalculate_matrices(self):
        M = nx.adjacency_matrix(self.G).toarray()
        self.adjacency_matrix = M
        M_sum = M.sum(axis=1).reshape((-1, 1))
        M_sum = np.where(np.isclose(M_sum, 0), np.inf, M_sum)
        self.stochastic_matrix = M / M_sum
        for i in range(len(M)):
            if M_sum[i] == np.inf:
                self.stochastic_matrix[i, i] = 1
        # print(self.stochastic_matrix)

    def r_in(self) -> np.ndarray:
        return self.adjacency_matrix.sum(axis=0)

    def r_out(self) -> np.ndarray:
        return self.adjacency_matrix.sum(axis=1)

    def one_limit_state(self) -> np.ndarray:
        if not nx.is_aperiodic(self.G):
            raise ValueError('Graph must be aperiodic for calculation of one limit state')
        n = len(self.adjacency_matrix)

        eigval, eigvect = sparce_eigs(self.stochastic_matrix.T, k=1, sigma=1.1)

        if not np.allclose(eigval, 1, atol=1e-7):
            raise RuntimeError(f'bad calculation of eigenvalue, it is {eigval}, not 1')
        if eigvect.shape == (n, 1):
            eigvect = np.real(eigvect.reshape(-1))
        else:
            raise RuntimeError(f'strange eigenvectors: {eigvect}')
        return eigvect / eigvect.sum()
        # state = np.zeros((n,))
        # state[0] = 1
        # state_prev = np.zeros((n,))
        # state_prev[-1] = 1
        # i = 0
        # while not np.allclose(state, state_prev, atol=1e-10) and i < 1000:
        #     state_prev, state = state, state @ self.stochastic_matrix
        #     i += 1
        # if i >= 1000:
        #     raise RuntimeError(f'Maximum iteration number exceeded, {state=}')
        # return state

    def T(self) -> float:
        q1_star = self.one_limit_state()
        r_out = self.r_out()
        T_i = r_out / q1_star
        min_ = T_i.min()
        return min_ 


    def _state_to_normal_form(self, q: Union[Dict[Node, float], List[float]]) -> Dict[Node, float]:
        if isinstance(q, dict):
            return q
        else:
            return {node: x for node, x in zip(self.node_descriptor.keys(), q)}
        

    def flow(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        q = q.reshape((-1,1))
        return np.minimum(q*self.stochastic_matrix, self.adjacency_matrix)
    
    def S(self, q: np.ndarray, flow=None) -> np.ndarray:
        q = np.asarray(q)
        flow = self.flow(q) if flow is None else flow 
        return q + flow.sum(axis=0) - flow.sum(axis=1)

    def __len__(self):
        return len(self.adjacency_matrix) 

    def add_weighted_edges_from(self, edge_bunch: Iterable[Tuple[Node, Node, float]]):
        def to_expected_form(it):
            return (it[0], it[1], {'weight': it[2]})
        self.G.add_edges_from(map(to_expected_form, edge_bunch))

        self.node_descriptor: dict[Node, int] = {node: i for i, node in enumerate(self.G.nodes)}
        self.idx_descriptor: list[Node] = [None]*len(self.G.nodes)
        for node, i in self.node_descriptor.items():
            self.idx_descriptor[i] = node
        self.recalculate_matrices()


    def run_simulation(self, initial_state: Union[Dict[Node, float], List[float]], n_iters=30)\
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

        state_dict = self._state_to_normal_form(initial_state)
        for j in range(n):
            state_arr[0, j] = state_dict[self.idx_descriptor[j]]
        for i in range(1, n_iters):
            flow_arr[i] = self.flow(state_arr[i-1])
            state_arr[i] = self.S(state_arr[i-1], flow=flow_arr[i])
        total_output_res: list[float] = [sum(map(lambda v: self.G[u][v]['weight'], self.G[u]))
                               for u in self.idx_descriptor]
        return StateArray(self.node_descriptor, self.idx_descriptor, state_arr, flow_arr, total_output_res)
    
    def plot_with_states(self, states: StateArray,
                        prop_setter: Optional[Callable[[nx.DiGraph], None]] = None,
                        scale=1.) -> Sequence[SVG]:
        G: nx.DiGraph = self.G.copy()
        res = [None]*len(states)
        
        G.graph['graph'] = {
            'layout': 'neato',
            'scale': scale
        }

        G.graph['node'] = {
            'fontsize': 10,
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': '#f0fff4',
            'fixedsize': True,
        }

        (prop_setter if prop_setter is not None else identity)(G)

        max_weight = max(map(lambda x: x[2]['weight'], G.edges(data=True)))
        min_weight = min(map(lambda x: x[2]['weight'], G.edges(data=True)))
        if np.allclose(max_weight, min_weight):
            calc_edge_width = lambda x: 2.5
        else:                        
            calc_edge_width = linear_func_from_2_points((min_weight, 0.8), (max_weight, 4.5))

        layout = nx.nx_pydot.pydot_layout(G, prog='neato') 
        layout_new = valmap(lambda x: (x[0]/45, x[1]/45), layout)
        void_node_dict = {}
        for v in G.nodes:
            G.nodes[v]['tooltip'] = str(v)
            pos = str(layout_new[v][0]) + ',' + str(layout_new[v][1]) + '!'
            G.nodes[v]['pos'] = pos
            void_node_dict[('void', v)] = {
                'pos': pos,
                'style': 'invis',
                'label': '',
                'color': 'transparent',
                'fillcolor': 'transparent',
                'tooltip': str(v),
                'width': MAX_NODE_WIDTH
                }
        G.add_nodes_from(void_node_dict.items())


        for u, v in G.edges:
            weight = self.G.edges[u, v]['weight']
            G.edges[u, v]['label'] = f'<<B>{weight}</B>>'
            G.edges[u, v]['penwidth'] = calc_edge_width(weight)
            G.edges[u, v]['arrowsize'] = 0.5
            G.edges[u, v]['fontsize'] = 14
            G.edges[u, v]['fontcolor'] = 'black'
            G.edges[u, v]['color'] = '#f3ad5c99'

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
        
        G.graph['graph'] = {
            'layout': 'neato',
            'scale': 1.7
        }

        for u, v in G.edges:
            G.edges[u, v]['label'] = G.edges[u, v]['weight']
        return SVG(nx.nx_pydot.to_pydot(G).create_svg())


class ResourceDiGraphWithIncome(ResourceDiGraph):
    def run_simulation(self, 
        initial_state: Union[Dict[Node, float], List[float]],
        n_iters=30,
        income_seq_func: Callable[[int], List[float]] = None)\
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

        if income_seq_func is None:
            income_seq_func = lambda t: [0]*n

        if isinstance(initial_state, dict):
            state_dict = initial_state
        else:
            state_dict = {node: x for node, x in zip(self.node_descriptor.keys(), initial_state)}
        for j in range(n):
            state_arr[0, j] = state_dict[self.idx_descriptor[j]]
        
        total_output_res: list[float] = [sum(map(lambda v: self.G[u][v]['weight'], self.G[u]))
                               for u in self.idx_descriptor]
        
        for i in range(1, n_iters):
            income_seq = income_seq_func(i-1)
            for u in self.G.nodes:
                u_i = self.node_descriptor[u]
                for v in self.G[u]:
                    v_i = self.node_descriptor[v]
                    transferred_res = min(
                        self.G[u][v]['weight']/total_output_res[u_i] * state_arr[i-1, u_i],
                        self.G[u][v]['weight'])
                    flow_arr[i, u_i, v_i] = transferred_res
                    state_arr[i, v_i] += transferred_res 
                state_arr[i, u_i] += max(state_arr[i-1, u_i] - total_output_res[u_i], 0) + income_seq[u_i]
                
        return StateArray(self.node_descriptor, self.idx_descriptor, state_arr, flow_arr, total_output_res)
    

def plot_simulation(G, simulation, scale=1.):
    pl = G.plot_with_states(simulation, scale=scale)
    f = lambda i: pl[i]
    interact(f, i=widgets.IntSlider(min=0,max=len(simulation)-1,step=1,value=0, description='№ of iteration')) 
    return None


def simple_protocol(sim: StateArray):
    n_iters = len(sim)
    vertices = lmap(get(sim.idx_descriptor), range(len(sim.idx_descriptor)))
    cols = ['t'] + vertices
    data = [None]*n_iters
    for i in range(n_iters):
        data[i] = [i] + lmap(get(sim[i]['states']), vertices)
    df = pd.DataFrame(columns=cols, data=data)
    return df.set_index('t')