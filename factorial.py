from os import stat
from typing import Dict
from operator import mul
from functools import reduce
from collections import Counter, defaultdict

# Factors = Dict[int, int]

class PossibleFactors(object):
    def __init__(self, n: int):
        self._N: int = n
    def __iter__(self):
        yield 2
        i=3
        while i*i < self._N+1:
            yield i
            i+=2

def factorise(n):
    factors = defaultdict(int)
    for factor in PossibleFactors(n):
        while n % factor == 0:
            n/=factor
            factors[factor]+=1
    if n != 1:
        factors[int(n)] += 1
    return factors

class Factor(object):
    def __init__(self, n=None):
        self._factors = defaultdict(int)
        if n is not None:
            self._factors = factorise(n)
    def __getitem__(self, f):
        return self._factors[f]
    def __setitem__(self, f, pow):
        self._factors[f] = pow

    def __mul__(self, other: 'Factor') -> 'Factor':
        g = Factor()
        for factor in Factor.combine_factor_dicts(self, other): #set().union(self._factors, other._factors): #type: ignore
            g[factor]=self[factor]+other[factor]
        return g

    def __truediv__(self, other: 'Factor') -> 'Factor':
        g = Factor()
        for factor in Factor.combine_factor_dicts(self, other): # set().union(self._factors, other._factors): #type: ignore
            g[factor]=self[factor]-other[factor]
        return g

    def eval(self):
        return reduce(mul,[key**val for key, val in self._factors.items()])
    def __str__(self):
        return ' x '.join(f'{key}^{val}' for key, val in self._factors.items())
    def __repr__(self):
        return str(self)

    @staticmethod
    def combine_factor_dicts(a,b):
        yield from set().union(a._factors, b._factors) #type: ignore

class Factorial(Factor):
    def __init__(self, n):
        self._factors = defaultdict(int)
        if n >= 2:
            for i in range(2,n):
                f = Factor(i)
                for factor, power in f._factors.items():
                    self._factors[factor] += power

def comb(n, k):
    a = Factorial(n)
    b = Factorial(k)
    c = Factorial(n-k)
    return a/(b*c)

def perm(n,k):
    a = Factorial(n)
    c = Factorial(n-k)
    return a/c

def multi(n,*args):
    n = Factorial(n)
    p = [Factorial(i) for i in args]
    P = reduce(mul, p, Factor())
    return n/P