import numpy as np
import pandas as pd
import sympy as sp
import networkx as nx
import pydot

import multiprocessing

import functools
from functools import reduce, partial
from toolz import *

import itertools
from itertools import product, starmap
import more_itertools
import operator

from IPython.display import Image, SVG
from ipywidgets import interact, widgets

from typing import Sequence, Callable, Tuple, List, Union, Iterable, Any, Hashable

#############################################################################################

inf = np.inf
get_curried = curry(operator.getitem)
lmap = compose(list, map)

def parallelize_range(n_pools, rng):
    rng = list(rng)
    total_len = len(rng)
    size_of_pool = total_len // n_pools + int(bool(total_len % n_pools))
    return partition_all(size_of_pool, rng)

def const_iter(x):
    while True:
        yield x

class PiecewiseLinearFunction:
    
    # in this class linear functions will be denoted
    LinearFunc = Tuple[float, float]
    
    Interval = Tuple[float, float]
    
    def __init__(self, x: List[float], f: List[LinearFunc]):
        assert len(x) + 1 == len(f),\
            ("Array sizes for constraining points (x) and " +
             "linear functions (f) should satisfy `len(x) + 1 == len(f), " +
             f"however len(x) == {len(x)}, len(f) == {len(f)}")
        self._x = x
        self._f = f
    
    @property
    def x(self): return self._x
    @property
    def f(self): return self._f
    
    @staticmethod
    def _is_close(x1: float, x2: float) -> bool:
        return abs(x1 - x2) < 1e-13
    
    @staticmethod
    def _linear_funcs_are_close(f1: LinearFunc, f2: LinearFunc) -> bool:
        return all(map(PiecewiseLinearFunction._is_close, f1, f2))
    
    def sanitize(self) -> 'PiecewiseLinearFunction':
        """
        Sometimes during function composition or addition
        we might get extra mesh points, like
        `f = (2*x | x <= 3), (2*x | 3 < x)`
        This function resolves it, thus f becomes simply
        `f = 2*x`
        """
        if any(starmap(PiecewiseLinearFunction._linear_funcs_are_close, 
                       sliding_window(2, self.f))):
            x, f = [], [self.f[0]]
            for i in range(len(self.x)):
                if not PiecewiseLinearFunction._linear_funcs_are_close(f[-1], self.f[i+1]):
                    x.append(self.x[i])
                    f.append(self.f[i+1])
            return PiecewiseLinearFunction(x, f)
        else:
            return self
    
    
    @staticmethod
    def _add_two(f1: LinearFunc, f2: LinearFunc):
        return (f1[0] + f2[0], f1[1] + f2[1])
    
    
    
    def merge_funcs(self,
            other: 'PiecewiseLinearFunction',
            merge_two: Callable[[LinearFunc, LinearFunc], LinearFunc])\
            -> 'PiecewiseLinearFunction':
        """
        res = merge_funcs(f1, f2, <op>) is similar to constructing 
        piecewise function, which mesh is sum of meshes of
        both f1 and f2. On each subdomain res is equal to
        `f1 <op> f2` on the according subdomains of f1 and f2
        """
        x, f = [],  []
        i = j = 0
        while i < len(self.x) and j < len(other.x):
            if PiecewiseLinearFunction._is_close(self.x[i], other.x[j]):
                x.append(self.x[i])
                f.append(merge_two(self.f[i], other.f[j]))
                i += 1
                j += 1                
            elif self.x[i] < other.x[j]:
                x.append(self.x[i])
                f.append(merge_two(self.f[i], other.f[j]))
                i += 1
            elif self.x[i] > other.x[j]:
                x.append(other.x[j])
                f.append(merge_two(self.f[i], other.f[j]))
                j += 1
        while i < len(self.x):
            x.append(self.x[i])
            f.append(merge_two(self.f[i], other.f[j]))
            i += 1
        while j < len(other.x):
            x.append(other.x[j])
            f.append(merge_two(self.f[i], other.f[j]))
            j += 1
        f.append(merge_two(self.f[i], other.f[j]))
        return PiecewiseLinearFunction(x, f).sanitize()
    
    def __add__(self, other: 'PiecewiseLinearFunction') -> 'PiecewiseLinearFunction':
        return self.merge_funcs(other, PiecewiseLinearFunction._add_two)
    
    def find_index(self, x: float) -> int:
        if len(self.x) == 0:
            res = 0
        elif x <= self.x[0]:
            res = 0
        elif self.x[-1] < x:
            res = len(self.x) # x cannot be indexed like that, though f can
        else:
            # simple binary search, actually
            l = 0
            r = len(self.x) - 1
            while r - l > 1:
                p = l + (r - l)//2
                if x <= self.x[p]:
                    r = p
                else:
                    l = p
            res = r
        return res
    
    @staticmethod
    def _inv_lin_if_possible(lin: LinearFunc) -> LinearFunc:
        if PiecewiseLinearFunction._is_close(lin[0], 0):
            raise ValueError(f"Cannot inverse constant function {lin}")
        else:
            return (1/lin[0], -lin[1]/lin[0])
    
    @staticmethod
    @curry
    def _apply_lin(lin: LinearFunc, x: float) -> float:
        if PiecewiseLinearFunction._is_close(lin[0], 0):
            return lin[1]
        else:
            return lin[0]*x + lin[1]
    
    @staticmethod
    def _interval_image(lin: LinearFunc, ival: Interval) -> Interval:
        # we assume `lin[0] != 0` 
        a, b = map(PiecewiseLinearFunction._apply_lin(lin), ival)
        return (a, b) if a < b else (b, a)
    
    def _points_and_idx_inside_interval(self, ival: Interval) -> Tuple[List[float], int]:
        ia, ib = map(self.find_index, ival)
        
        if ia == len(self.x):
            return ([], ia)
        elif ib == len(self.x):
            ia_corr = ia if not PiecewiseLinearFunction._is_close(ival[0], self.x[ia])\
                        else ia + 1
            ib_corr = ib
            return (self.x[ia_corr:ib_corr], ia_corr)
        else:
            ia_corr = ia if not PiecewiseLinearFunction._is_close(ival[0], self.x[ia])\
                        else ia + 1
            ib_corr = ib if not PiecewiseLinearFunction._is_close(ival[1], self.x[ib])\
                        else ib - 1
            if ib_corr == -1: 
                return ([], ia_corr)
            return (self.x[ia_corr:ib_corr], ia_corr)
    
    @staticmethod
    def _inverse_points(lin: LinearFunc, points: List[float]) -> List[float]:
        lin_inv = PiecewiseLinearFunction._inv_lin_if_possible(lin)
        if lin[0] > 0:
            return lmap(PiecewiseLinearFunction._apply_lin(lin_inv), points)
        else:
            # since we don't have efficient inversed map, making a plain loop here:
            res = [None]*len(points)
            for i, p in enumerate(points):
                res[- 1 - i] = PiecewiseLinearFunction._apply_lin(lin_inv, p)
            return res
#         return (reversed if lin[0] < 0 else identity)(lmap(PiecewiseLinearFunction._apply_lin(lin_inv), points))

    def partition_domain(self, f2: LinearFunc, ival: Interval)\
        -> Tuple[List[float], int]:
        """
        Provided `f2` has domain `ival` and we are to calculate `compose(f1, f2)`,
        returns a proper partition for `ival` (makes a corresponding mesh from it).
        !!! Beware, result's points don't contain `ival[0]` and `ival[1]` !!! 
        """
        #in case of constant function `f2` we're not extending the domain:
        if PiecewiseLinearFunction._is_close(f2[0], 0):
            idx = self.find_index(PiecewiseLinearFunction._apply_lin(f2, ival[1]))
            return ([], idx)
        else:
            im = PiecewiseLinearFunction._interval_image(f2, ival)
            points, idx = self._points_and_idx_inside_interval(im)
            inv_points = PiecewiseLinearFunction._inverse_points(f2, points)
            return (inv_points, idx)
    
    @staticmethod
    def _compose_lin(f1: LinearFunc, f2: LinearFunc) -> LinearFunc:
        return (f1[0]*f2[0], f1[0]*f2[1] + f1[1])
    
    @staticmethod
    def compose(f1: 'PiecewiseLinearFunction', f2: 'PiecewiseLinearFunction')\
            -> 'PiecewiseLinearFunction':
        
        def process(rhs_lin_func, ival):
            new_points, f_idx = f1.partition_domain(rhs_lin_func, ival)
            if PiecewiseLinearFunction._is_close(rhs_lin_func[0], 0) or rhs_lin_func[0] > 0:
                idx = f_idx
                new_idx = lambda old: old + 1
            else:
                idx = len(new_points) + f_idx
                new_idx = lambda old: old - 1
                
            for p in new_points:
                f.append(PiecewiseLinearFunction._compose_lin(f1.f[idx], rhs_lin_func))
                idx = new_idx(idx)
                x.append(p)
            f.append(PiecewiseLinearFunction._compose_lin(f1.f[idx], rhs_lin_func))
        
        x, f = [], []
        if len(f2.x) == 0:
            x = f1.x
            f = lmap(lambda it: PiecewiseLinearFunction._compose_lin(it, f2.f[0]),
                     f1.f)
        else:            
            # case (-inf, f2.x[0]):
            process(f2.f[0], (-inf, f2.x[0]))
            x.append(f2.x[0])
            for i in range(len(f2.x) - 1):
                process(f2.f[i+1], (f2.x[i], f2.x[i+1]))
                x.append(f2.x[i+1])
            #case (f2.x[-1], +inf):
            process(f2.f[-1], (f2.x[-1], +inf))
        
        return PiecewiseLinearFunction(x, f).sanitize()
        
    
    def __call__(self, x: float):
        k, b = self.f[self.find_index(x)]
        return k*x + b
    
    def to_str(self):
        def conv_f(f):
            if PiecewiseLinearFunction._is_close(f[0], 0):
                return '{:.6g}'.format(f[1])
            elif PiecewiseLinearFunction._is_close(f[1], 0):
                return '{:.6g}*x'.format(f[0])
            else:
                return '{:.6g}*x + {:.6g}'.format(f[0], f[1])
        def conv_x(x):
            return '{:.6g}'.format(x)
        
#         "(2*x + 3 | 0 < x <= 1), (2 | 1 < x <= 3)"
        
        if len(self.x) == 0:
            return conv_f(self.f[0])
        else:
            s0 =  f'({conv_f(self.f[0])} | x <= {conv_x(self.x[0])})'
            ls = (f'({conv_f(self.f[i])} | {conv_x(self.x[i-1])} < x <= {conv_x(self.x[i])})'
                      for i in range(1, len(self.x)))
            sn =  f'({conv_f(self.f[-1])} | {conv_x(self.x[-1])} < x)'
            
            
        return ', '.join(itertools.chain([s0], ls, [sn]))
    
    def __repr__(self):
        return 'Piecewise linear: ' +  self.to_str()

    
def resource_piecewise(transition, limitation):
    x = [0, limitation]
    f = [(0, 0), (transition/limitation, 0), (0, transition)]
    return PiecewiseLinearFunction(x, f)

def self_resource_piecewise(capacity):
    x = [capacity]
    f = [(0, 0), (1, -capacity)]
    return PiecewiseLinearFunction(x, f)

pw_compose = PiecewiseLinearFunction.compose


class PLF_AlgebraElement:
    
    Zero : 'PLF_AlgebraElement'
    One :  'PLF_AlgebraElement'

    def __init__(self, pwf: PiecewiseLinearFunction):
        self.pwf = pwf
    
    def __add__(self, other):
        return PLF_AlgebraElement(self.pwf + other.pwf)
    def __mul__(self, other):
        return PLF_AlgebraElement(PiecewiseLinearFunction.compose(self.pwf, other.pwf))
    def __call__(self, x):
        return self.pwf(x)

    def __repr__(self):
        return 'PLF_Alg: <' +  self.pwf.to_str() + '>'

PLF_AlgebraElement.Zero = PLF_AlgebraElement(PiecewiseLinearFunction([], [(0, 0)]))
PLF_AlgebraElement.One  = PLF_AlgebraElement(PiecewiseLinearFunction([], [(1, 0)]))

resource_alg  = compose(PLF_AlgebraElement, resource_piecewise)
resource_self = compose(PLF_AlgebraElement, self_resource_piecewise)


class ApplicativeMatrix:
    
    def __init__(self, matr: np.ndarray):
        self.matr = matr if isinstance(matr, np.ndarray) else np.array(matr, dtype=object) 
        self.type = None if self.matr.shape == (0, 0) else type(self.matr[0,0])
        
        def types_similar():
            for (i, j), x in np.ndenumerate(self.matr):
                yield(type(self.matr[i,j]) is self.type)

        assert all(types_similar()), \
            'Matrix elements should be of homogeneous type'
    
    
    def __matmul__(self, other: 'ApplicativeMatrix'):
        return ApplicativeMatrix(self.matr@other.matr)
    
    def __pow__(self, n):
        if n < 0:
            raise ValueError('Applicative matrix cannot be inversed')
        elif n == 0:
            if self.type is PLF_AlgebraElement:
                m = np.full(self.matr.shape, PLF_AlgebraElement.Zero, dtype=object)
                np.fill_diagonal(m, PLF_AlgebraElement.One)
                return m
            else:
                ApplicativeMatrix(np.linalg.matrix_power(self.matr, n))
        else:
            return ApplicativeMatrix(np.linalg.matrix_power(self.matr, n))
    
    def __call__(self, x: Union[List[float], np.ndarray]):
        if len(x) != self.matr.shape[1]:
            raise ValueError('Dimensions must agree')
        res = np.full((self.matr.shape[0],), None, dtype=object)
        for i in range(self.matr.shape[0]):
            res[i] = reduce(operator.add, map(lambda f, x: f(x), self.matr[i], x))
        return res
    
    def __setitem__(self, idx, val):
        self.matr[idx] = val
    
    def __repr__(self):
        return 'ApplicativeMatrix: ' +  repr(self.matr)  


Node = Hashable

class StateArray:
    def __init__(self, idx_descriptor: dict[int, Node], arr: np.ndarray):
        self.idx_descriptor = idx_descriptor
        self.arr = arr
#         self.reversed_node_descriptor = {v: k for k, v in node_descriptor}
    
    def __len__(self) -> int:
        return len(self.arr)
    
    def __getitem__(self, time) -> dict[Node, float]:
        t_arr = self.arr[time]
        return {node : t_arr[idx] for idx, node in self.idx_descriptor.items()}


def remove_edge_if_possible(G, edge):
    if edge in G.edges:
        G.remove_edge(*edge)



def calc_in_range(G, states, rng):
    res = [None]*len(rng)
    n_it = 0
    for idx in rng:
        for v in G.nodes:            
            G.nodes[v]['label'] = f"""<<table>
                                        <tr><td>{v}</td></tr>
                                        <tr><td bgcolor='#00CC11'>{'{:.4g}'.format(states[idx][v])}</td></tr>
                                        </table>>"""
        res[n_it] = SVG(nx.nx_pydot.to_pydot(G).create_svg())            
        n_it += 1
    return res

class ResourceDiGraph:
    
    def __init__(self):
        self._graph_is_vaild = False
        self._G = nx.DiGraph()
        self._applicative_matrix: ApplicativeMatrix = None
        self._node_descriptor: dict[Node, int] = {}
        self.edge_capacities: dict[Node, dict[Node, float]] = {}
    
    def _ensure_invariants(self):
        self._graph_is_vaild = True
        n = len(self._G.nodes)
        self._node_descriptor = dict(zip(self._G.nodes, itertools.count(0)))
        
        for v in self._G.nodes:
            if (v, v) not in self._G.edges:
                self._G.add_edge(v, v, resource_func=resource_self(0))

        self._applicative_matrix = ApplicativeMatrix(
            np.full((n, n), PLF_AlgebraElement.Zero, dtype=object))
        for u, v in self._G.edges:
            self._applicative_matrix[
                self.node_descriptor[v],
                self.node_descriptor[u]
            ] = self._G[u][v]['resource_func']
    
    def _ensure_and_return(self, property_to_return: str) -> Any:
        if not self._graph_is_vaild:
            self._ensure_invariants()
        return self.__getattribute__(property_to_return)

    # !!! Attention !!! Applicative matrix is a *transposed* adjacency matrix
    @property
    def applicative_matrix(self) -> ApplicativeMatrix:
        return self._ensure_and_return('_applicative_matrix')

    @property
    def node_descriptor(self) -> dict[Node, int]:
        return self._ensure_and_return('_node_descriptor')

    @property
    def G(self) -> nx.DiGraph:
        return self._ensure_and_return('_G')

    def add_outedges_of_node(
            self,
            node: Node,
            edge_capacities: dict[Node, float]) -> 'ResourceDiGraph':

        integral_capacity = sum(edge_capacities.values())
        for v, cap in edge_capacities.items():
            remove_edge_if_possible(self._G, (node, v))
            self._G.add_edge(node, v, resource_func=resource_alg(cap, integral_capacity),
                            label=cap)
        remove_edge_if_possible(self._G, (node, node))
        self._G.add_edge(node, node, resource_func=resource_self(integral_capacity))
        
        self._graph_is_vaild = False
        self.edge_capacities[node] = edge_capacities
        return self

    def run_simulation(self, initial_state: dict[Node, float], n_iters=20)\
        -> StateArray:
        if len(initial_state) != len(self._G.nodes):
            raise ValueError(
                'Incorrect initial states: expected states for ' +
                str(self._G.nodes) +
                ', while got:' +
                str(list(initial_state.keys())))
        n = len(initial_state)
        idx_descriptor = {v: k for k, v in self.node_descriptor.items()}
        res = np.zeros((n_iters, n))
        for j in range(n):
            res[0, j] = initial_state[idx_descriptor[j]]
        for i in range(1, n_iters):
            res[i] = self.applicative_matrix(res[i-1])
        return StateArray(idx_descriptor, res)
    
    def plot_with_states(self, states: StateArray) -> list[SVG]: # np.ndarray:
        G = self._G.copy()
        res = [None]*len(states)
        for v in G.nodes:
            G.nodes[v]['shape'] = 'plaintext'
            if (v, v) in G.edges:
                G.remove_edge(v, v)
        for u, v in G.edges:
            G.edges[u, v]['label'] = self.edge_capacities[u][v]

        # for i in range(len(res)):
        #     for v in G.nodes:            
        #         G.nodes[v]['label'] = f"""<<table>
        #                                        <tr><td>{v}</td></tr>
        #                                        <tr><td bgcolor='#00CC11'>{'{:.4g}'.format(states[i][v])}</td></tr>
        #                                     </table>>"""
        #     res[i] = SVG(nx.nx_pydot.to_pydot(G).create_svg())
        # return res

        n_pools = min(8, len(states.arr))
        pool_obj = multiprocessing.Pool(n_pools)
        answer = pool_obj.starmap(
            calc_in_range,
            zip(const_iter(G), const_iter(states), parallelize_range(n_pools, range(len(res)))))
        return np.concatenate(answer)
    

    def plot(self):
        G = self._G.copy()
        for v in G.nodes:
            if (v, v) in G.edges:
                G.remove_edge(v, v)
        for u, v in G.edges:
            G.edges[u, v]['label'] = self.edge_capacities[u][v]
        return SVG(nx.nx_pydot.to_pydot(G).create_svg())

# G = ResourceDiGraph()
# G.add_outedges_of_node(0, {1: 3, 2: 4})
# G.add_outedges_of_node(1, {0: 2, 2: 6})
# sim = G.run_simulation({0: 30, 1: 50, 2: 1}, 5)