import numpy as np
import pandas as pd

import sympy as sp

import functools
from functools import reduce, partial
from toolz import * #curry, compose, compose_left as pipe
from enum import Enum

import itertools
import more_itertools

from itertools import product

# from abc import *

import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["axes.grid"] = True

plt.rcParams["font.size"] = 12

import operator

from typing import Sequence, Callable, Tuple, List



inf = np.inf
get_curried = curry(operator.getitem)
lmap = compose(list, map)

# class PiecewiseFunction(ABC):
    
#     @property
#     @abstractmethod
#     def x(self): pass
#     @property
#     @abstractmethod
#     def f(self): pass
    
#     @staticmethod
#     @abstractmethod
#     def merge_two(f1, f2): pass
    
#     def merge_funcs(self, other):
#         assert type(self) == type(other)
#         res = type(self)([], [])
#         i = j = 0
        
        
###################################################################################################

class PiecewiseLinearFunction:
    pass

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
    def _add_two(f1: LinearFunc, f2: LinearFunc):
        return (f1[0] + f2[0], f1[1] + f2[1])
    
    def merge_funcs(self,
            other: PiecewiseLinearFunction,
            merge_two: Callable[[LinearFunc, LinearFunc], LinearFunc])\
            -> PiecewiseLinearFunction:
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
    
    def __add__(self, other: PiecewiseLinearFunction) -> PiecewiseLinearFunction:
        return self.merge_funcs(other, PiecewiseLinearFunction._add_two)
    
    def find_index(self, x: float) -> int:
        if x <= self.x[0]:
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
        x, f = [], [self.f[0]]
        for i in range(len(x)):
            if not PiecewiseLinearFunction._linear_funcs_are_close(f[-1], self.f[i+1]):
                x.append(self.x[i])
                f.append(self.f[i+1])
        return PiecewiseLinearFunction(x, f)
        
        
    
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
    def compose(f1: PiecewiseLinearFunction, f2: PiecewiseLinearFunction)\
            -> PiecewiseLinearFunction:
        
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

    
def resource_piecewise(limitation, transition):
    x = [0, limitation]
    f = [(0, 0), (transition/limitation, 0), (0, transition)]
    return PiecewiseLinearFunction(x, f)

    
pw_compose = PiecewiseLinearFunction.compose


p1, p2 = resource_piecewise(1, 4), resource_piecewise(3, 5)
comp = pw_compose(p1, p2)

