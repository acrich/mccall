from interpolation import interp
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt
from numba import njit, prange, float64, int64
import quantecon as qe


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)





try:
    v = np.load('v.npy')
    h = np.load('h.npy')
    accept_or_reject = np.load('accept_or_reject.npy')
    a_opt_unemployed = np.load('a_opt_unemployed.npy')
    a_opt_employed = np.load('a_opt_employed.npy')
except IOError:
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = solve_model()
    np.save('v.npy', v)
    np.save('h.npy', h)
    np.save('accept_or_reject.npy', accept_or_reject)
    np.save('a_opt_unemployed.npy', a_opt_unemployed)
    np.save('a_opt_employed.npy', a_opt_employed)


print(accept_or_reject)
