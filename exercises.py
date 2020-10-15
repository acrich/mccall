import numpy as np
import time
import quantecon as qe
from numba import jit
import matplotlib.pyplot as plt
from random import uniform
import sympy as sym
from sympy import init_printing
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def numpy_ex_1():
    def p(x, coeff):
        arr = np.ones(len(coeff))
        arr[1:] = x
        arr = arr.cumprod()
        return coeff @ arr

    coef = np.array([2,4,6])
    x = 3
    print(p(x, coef))
    print(np.poly1d(np.flip(coef))(x))



def numpy_ex_2():
    class DiscreteRV(object):
        def __init__(self, pmf):
            """ pmf is a probability mass function for the discrete random variable """
            self.q = np.asarray(pmf)
            self.cq = self.q.cumsum()

        def sample(self, U):
            """ returns the index in the pmf of a single draw """
            a = 0.0
            for i in range(len(self.q)):
                if a < U <= a + self.q[i]:
                    return i
                a = a + self.q[i]

        def draw(self, U):
            """ same as sample, only efficient using vectorization """

            return self.cq.searchsorted(U)


    rv = DiscreteRV([0.25, 0.75])
    draws = [uniform(0, 1) for i in range(10000)]
    start = time.time()
    a = [rv.sample(u) for u in draws]
    print(time.time() - start)
    start = time.time()
    a = [rv.draw(u) for u in draws]
    print(time.time() - start)


def matplotlib_ex_1():
    fig, ax = plt.subplots()
    θs = np.linspace(0, 2, 10)
    x = np.linspace(0, 5, 150)
    def f(θ, x):
        return np.cos(np.pi*θ*x)*np.exp(-x)
    for θ in θs:
        f = np.vectorize(f)
        ax.plot(x, f(θ, x), linewidth=2, alpha=0.6)
    plt.show()


def markov(n):
    x = np.empty(n, dtype=np.int_)
    x[0] = 1
    low_cdf = np.array([0.9, 0.1], dtype=np.float32).cumsum()
    high_cdf = np.array([0.2, 0.8], dtype=np.float32).cumsum()
    for t in range(1, n):
        u = uniform(0, 1)
        if x[t-1] == 0:
            x[t] = np.searchsorted(low_cdf, u)
        else:
            x[t] = np.searchsorted(high_cdf, u)
    return (n - np.sum(x))/n

qe.tic()
print(markov(1000000))
print(qe.toc())

markov_numba = jit(markov)

qe.tic()
print(markov_numba(1000000))
qe.toc()
