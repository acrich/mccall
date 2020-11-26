import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import quantecon as qe
from interpolation import interp
from numpy.random import randn
from numba import njit, prange, float64, int32
from numba.experimental import  jitclass


def binomial_draws(n=1000, α=0.1, seed=None):
    """ returns n draws from binomial distribution with probability α """
    if seed is None:
        seed = random.randint(1000,10001)
    np.random.seed(seed)
    draws = bernoulli.rvs(1 - α, size=n)
    return draws


def plot_it():
    """ plot separations from binomial distribution over a life span of T periods """
    T = 100
    separation_rate = 0.5
    seed = 2345
    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.set_ylabel('utility')

    stays_employed = binomial_draws(n=T, α=separation_rate, seed=seed)
    ax.plot(range(T), stays_employed, '--', alpha=0.4, label=f"$\alpha$ = {separation_rate}")
    plt.show()


if __name__ == '__main__':
    plot_it()
