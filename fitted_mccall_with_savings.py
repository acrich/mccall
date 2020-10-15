import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import quantecon as qe
from interpolation import interp
from numpy.random import randn
from numba import njit, prange, float64, int32
from numba.experimental import  jitclass


np.set_printoptions(threshold=sys.maxsize)


"""
Overview:
We've taken the basic McCall model, as implemented in quantecon.org, and plan
to apply the following steps:
1. add separations (following example in quantecon.org)
2. add continuous distribution of w (again, following quantecon.org)
3. add savings. from now on I'm by myself.
4. add multiple agents with heterogeneity in a_0
5. add w that's resolved competitively using a technology that includes a stochastic TFP.

With savings allowed, the equations from:
https://python.quantecon.org/mccall_model_with_separation.html#The-Bellman-Equations
become:

v(w_e, a) = u(c_e) + \beta [(1 - \alpha) v(w_e, a') + \alpha \sum_{w'} h(w', a') q(w')] \; s.t. \; c_e + a' = a + w_e
h(w, a) = \max \{ v(w, a), u(c_u) + \beta \sum_{w'} h(w', a') q(w') \} \; s.t. \; c_u + a' = a

and equations (5) and (6) become:

d(a') = \sum_{w'} \{ v(w', a'), u(a' - a'') +\beta d(a'') \} q(w')
v(w,a) = u(a + w - a') + \beta [(1 - \alpha) v(w, a') + \alpha d(a')]
"""


def binomial_draws(n=1000, α=0.1, seed=2345):
    np.random.seed(seed)
    draws = bernoulli.rvs(1 - α, size=n)
    return draws


def lognormal_draws(n=100, μ=2.5, σ=0.5, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws


# mccall_data_continuous = [
#     ('c', float64),          # unemployment compensation
#     ('α', float64),          # job separation rate
#     ('β', float64),          # discount factor
#     ('σ', float64),          # scale parameter in lognormal distribution
#     ('μ', float64),          # location parameter in lognormal distribution
#     ('a', float64),          # current savings rate
#     ('a_grid', float64[:]),  # grid of savings points for regular VFI
#     ('w_grid', float64[:]),  # grid of wage points for fitted VFI
#     ('w_draws', float64[:]), # draws of wages for Monte Carlo
#     ('w_grid_size', int32),  # number of points in wage grid
#     ('a_grid_size', int32),  # number of points in savings grid
# ]


class McCallModelContinuous:

    def __init__(self,
                 c=1,
                 α=0.1,
                 β=0.96,
                 grid_min=1e-10,
                 grid_max=50,
                 a_grid_max=100,
                 a_grid_min=100,
                 a_grid_size=1,
                 grid_size=50,
                 a_0=100,
                 w_draws=lognormal_draws()):

        self.c, self.α, self.β = c, α, β
        self.a = a_0

        self.w_grid = np.linspace(grid_min, grid_max, grid_size)
        self.w_draws = w_draws

        self.w_grid_size = grid_size
        self.a_grid_size = a_grid_size
        self.a_grid_min = a_grid_min
        self.a_grid_max = a_grid_max

        self.a_grid = np.linspace(self.a_grid_min, self.a_grid_max, self.a_grid_size)

    def update(self, v, d):
        # Simplify names
        c, α, β = self.c, self.α, self.β
        u = lambda x: np.log(x)

        # Interpolate array represented value function
        vf = []
        for i in range(self.a_grid_size):
            vf.append(lambda x: interp(self.w_grid, v[i, :], x))

        a_opt_employed = np.empty((self.a_grid_size, self.w_grid_size), dtype=int)
        a_opt_unemployed = np.empty_like(self.a_grid, dtype=int)

        v_new = np.empty((self.a_grid_size, self.w_grid_size), dtype=np.float64)
        d_new = np.empty_like(self.a_grid, dtype=np.float64)

        # Update d using Monte Carlo to evaluate integral
        for i in range(self.a_grid_size):
            # right-hand of the RHS means we look at every possible a', and find the one
            # that maximizes utility (VFI). The choice of a' isn't recorded anywhere.
            # The left-hand side is a McCall-style mean over all possible future w (fitted VFI).
            # finally, we make the outer {accept,reject} maximization choice.
            rhs = u(c + self.a_grid[i] - self.a_grid) + β * d
            d_new[i] = np.mean(np.maximum(vf[i](self.w_draws), np.nanmax(rhs)))
            a_opt_unemployed[i] = np.nanargmax(rhs)

        # Update v
        for i in range(self.a_grid_size):
            for j in range(self.w_grid_size):
                rhs = u(self.a_grid[i] - self.a_grid + self.w_grid[j]) + β * ((1 - α) * v[:, j] + α * d)
                v_new[i, j] = np.nanmax(rhs)
                a_opt_employed[i, j] = np.nanargmax(rhs)

        return v_new, d_new, a_opt_employed, a_opt_unemployed


# We then return the current iterate as an approximate solution.

def solve_model(mcm, tol=1e-5, max_iter=200):
    """
    Iterates to convergence on the Bellman equations

    * mcm is an instance of McCallModel
    """

    v = np.ones((mcm.a_grid_size, mcm.w_grid_size))     # Initial guess of v
    d = np.ones_like(mcm.a_grid)                        # Initial guess of d
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, d_new, a_opt_employed, a_opt_unemployed = mcm.update(v, d)
        error_1 = np.max(np.max(np.abs(v_new - v)))
        error_2 = np.max(np.abs(d_new - d))
        error = max(error_1, error_2)
        v = v_new
        d = d_new
        i += 1

    return v, d, a_opt_employed, a_opt_unemployed


def draw_lifetime(mcm, T=100):
    u_t = np.empty(T)
    u = lambda x: np.log(x)

    v, d, a_opt_employed, a_opt_unemployed = solve_model(mcm)
    print(a_opt_employed)
    print(a_opt_unemployed)

    accept_or_reject = np.empty((mcm.a_grid_size, mcm.w_grid_size))
    h = np.empty((mcm.a_grid_size, mcm.w_grid_size))
    # for j in range(mcm.w_grid_size):
    #     accept, reject = v[:, j], u(mcm.c) + mcm.β * d
    #     h[:, j] = np.maximum(accept, reject)
    #     # get indices for which array each max value came from.
    #     # see: https://stackoverflow.com/a/26900483/1408861
    #     single_2d_array = np.vstack((accept, reject))
    #     accept_or_reject[:, j] = np.argmax(single_2d_array, axis=0)
    for i in range(mcm.a_grid_size):
        for j in range(mcm.w_grid_size):
            accept = v[i, j]
            reject = np.max(u(mcm.c + mcm.a_grid[i] - mcm.a_grid) + mcm.β * d)
            h[i, j] = np.nanmax([accept, reject])
            accept_or_reject[i, j] = np.argmax(np.array([reject, accept]))

    w = lognormal_draws(n=T)

    vf = []
    for i in range(mcm.a_grid_size):
        vf.append(lambda x: interp(mcm.w_grid, v[i, :], x))
    hf = []
    for i in range(mcm.a_grid_size):
        hf.append(lambda x: interp(mcm.w_grid, h[i, :], x))

    path = []
    is_employed = 1
    a = np.argmax(mcm.a_grid >= mcm.a)
    a_t = np.empty(T)
    stays_employed = binomial_draws(n=T)
    for t in range(T):
        a_t[t] = mcm.a_grid[a]
        w_t_on_grid = np.argmax(mcm.w_grid >= w[t])
        if is_employed:
            u_t[t] = vf[a](w[t])
            a = a_opt_employed[a, w_t_on_grid]
            path.append((t, w[t], w_t_on_grid, u_t[t], 1, is_employed))
            is_employed = stays_employed[t]
        else:
            is_employed = accept_or_reject[a, w_t_on_grid]
            if is_employed:
                u_t[t] = vf[a](w[t])
                path.append((t, w[t], w_t_on_grid, u_t[t], 2, is_employed))
                a = a_opt_employed[a, w_t_on_grid]
            else:
                u_t[t] = hf[a](w[t])
                path.append((t, w[t], w_t_on_grid, u_t[t], 3, is_employed))
                a = a_opt_unemployed[a]

    return u_t, a_t, path


def plot_lifetime():
    T = 200
    mcm = McCallModelContinuous(
        a_grid_max=400,
        a_grid_min=1e-10,
        a_grid_size=100
    )
    u_t, a_t, path = draw_lifetime(mcm, T)
    unemployment_spells = []
    for t in range(T):
        unemployment_spells.append((path[t][4], path[t][5]))

    print(unemployment_spells)

    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.set_ylabel('utility')

    ax.plot(range(T), u_t, '--', alpha=0.4, label=f"$u(w_t)$")
    ax.plot(range(T), a_t, '-', alpha=0.4, label=f"$a_t$")
    plt.show()

if __name__ == '__main__':
    plot_lifetime()
