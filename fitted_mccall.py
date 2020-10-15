import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import quantecon as qe
from interpolation import interp
from numpy.random import randn
from numba import njit, prange, float64, int32
from numba.experimental import  jitclass


def binomial_draws(n=1000, α=0.1, seed=2345):
    np.random.seed(seed)
    draws = bernoulli.rvs(1 - α, size=n)
    return draws


def lognormal_draws(n=100, μ=2.5, σ=0.5, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws


mccall_data_continuous = [
    ('c', float64),          # unemployment compensation
    ('α', float64),          # job separation rate
    ('β', float64),          # discount factor
    ('σ', float64),          # scale parameter in lognormal distribution
    ('μ', float64),          # location parameter in lognormal distribution
    ('w_grid', float64[:]),  # grid of points for fitted VFI
    ('w_draws', float64[:])  # draws of wages for Monte Carlo
]


class McCallModelContinuous:

    def __init__(self,
                 c=1,
                 α=0.1,
                 β=0.96,
                 grid_min=1e-10,
                 grid_max=50,
                 grid_size=100,
                 w_draws=lognormal_draws()):

        self.c, self.α, self.β = c, α, β

        self.w_grid = np.linspace(grid_min, grid_max, grid_size)
        self.w_draws = w_draws

    def update(self, v, d):
        # Simplify names
        c, α, β = self.c, self.α, self.β
        w = self.w_grid
        u = lambda x: np.log(x)

        # Interpolate array represented value function
        vf = lambda x: interp(w, v, x)

        # Update d using Monte Carlo to evaluate integral
        d_new = np.mean(np.maximum(vf(self.w_draws), u(c) + β * d))

        # Update v
        v_new = u(w) + β * ((1 - α) * v + α * d)

        return v_new, d_new


# We then return the current iterate as an approximate solution.

def solve_model(mcm, tol=1e-5, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations

    * mcm is an instance of McCallModel
    """

    v = np.ones_like(mcm.w_grid)    # Initial guess of v
    d = 1                           # Initial guess of d
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, d_new = mcm.update(v, d)
        error_1 = np.max(np.abs(v_new - v))
        error_2 = np.abs(d_new - d)
        error = max(error_1, error_2)
        v = v_new
        d = d_new
        i += 1

    return v, d


# Here’s a function compute_reservation_wage that takes an instance of McCallModelContinuous and returns the associated reservation wage.
# If v(w)<h for all w, then the function returns np.inf

def compute_reservation_wage(mcm):
    """
    Computes the reservation wage of an instance of the McCall model
    by finding the smallest w such that v(w) >= h.

    If no such w exists, then w_bar is set to np.inf.
    """
    u = lambda x: np.log(x)

    v, d = solve_model(mcm)
    h = u(mcm.c) + mcm.β * d

    w_bar = np.inf
    for i, wage in enumerate(mcm.w_grid):
        if v[i] > h:
            w_bar = wage
            break

    return w_bar


def plot_results(mcm):
    """
    plots h(w) and v(w) for every w using an instance of
    the McCall model.
    """
    u = lambda x: np.log(x)

    fig, ax = plt.subplots()
    ax.set_xlabel('wage')
    ax.set_ylabel('value')

    v, d = solve_model(mcm)
    h = np.maximum(v, u(mcm.c) + mcm.β * d)

    ax.plot(mcm.w_grid, v, '-', alpha=0.4, label=f"v(w)")
    ax.plot(mcm.w_grid, h, '--', alpha=0.4, label=f"h(w)")
    ax.legend(loc='lower right')
    plt.show()


def draw_lifetime(mcm, T=100):
    u_t = np.empty(T)
    v, d = solve_model(mcm)
    u = lambda x: np.log(x)
    h = np.maximum(v, u(mcm.c) + mcm.β * d)
    rej = np.ones(100) * (u(mcm.c) + mcm.β * d)
    accept_or_reject = np.argmax(np.vstack((v, rej)), axis=0)
    w = lognormal_draws(n=T)
    vf = lambda x: interp(mcm.w_grid, v, x)
    hf = lambda x: interp(mcm.w_grid, h, x)
    stays_employed = binomial_draws()
    is_employed = 1

    path = []
    for t in range(T):
        w_t_on_grid = np.argmax(mcm.w_grid >= w[t])
        if is_employed:
            u_t[t] = vf(w[t])
            path.append((t, w[t], w_t_on_grid, u_t[t], 1))
            is_employed = stays_employed[t]
        else:
            is_employed = accept_or_reject[w_t_on_grid]
            if is_employed:
                u_t[t] = vf(w[t])
                path.append((t, w[t], w_t_on_grid, u_t[t], 2))
            else:
                u_t[t] = hf(w[t])
                path.append((t, w[t], w_t_on_grid, u_t[t], 3))

    return u_t, path


def plot_lifetime():
    T = 100
    mcm = McCallModelContinuous()
    u_t = draw_lifetime(mcm, T)

    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.set_ylabel('utility')

    ax.plot(range(T), u_t, '--', alpha=0.4, label=f"$u(w_t)$")
    plt.show()


mcm = McCallModelContinuous()
plot_results(mcm)
print(compute_reservation_wage(mcm))
