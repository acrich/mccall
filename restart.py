import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import quantecon as qe
from interpolation import interp
from numpy.random import randn
from numba import njit, prange, float64, int32
from numba.experimental import  jitclass
from separations import binomial_draws


def lognormal_draws(n=100, μ=2.5, σ=0.5, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws


class BaseMcCall:

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


class McCallWithSavings(BaseMcCall):

    def __init__(self,
                 c=1,
                 α=0.1,
                 β=0.96,
                 grid_min=1e-10,
                 grid_max=50,
                 grid_size=100,
                 a_grid_min=1e-10,
                 a_grid_max=50,
                 a_grid_size=100,
                 w_draws=lognormal_draws()):

        self.c, self.α, self.β = c, α, β

        self.w_grid = np.linspace(grid_min, grid_max, grid_size)
        self.a_grid = np.linspace(a_grid_min, a_grid_max, a_grid_size)
        self.w_draws = w_draws

    def get_next_d(self, v, d):
        c, α, β = self.c, self.α, self.β
        u = lambda x: np.log(x)

        # Interpolate array represented value function
        vf = [x for x in np.zeros_like(self.a_grid)]
        for i in range(len(self.a_grid)):
            vf[i] = lambda x: interp(self.w_grid, v[i, :], x)

        # Update d using Monte Carlo to evaluate integral
        a_opt_unemployed = np.empty_like(self.a_grid)
        for i in range(len(self.a_grid)):
            rhs = u(c + self.a_grid[i] - self.a_grid) + β * d
            a_opt_unemployed[i] = np.nanargmax(rhs)
            d_new[i] = np.mean(np.maximum(vf[i](self.w_draws), np.nanmax(rhs)))
        return d_new

    def get_next_v(self, v, d):
        c, α, β = self.c, self.α, self.β
        u = lambda x: np.log(x)

        # Update v
        v_new = np.empty_like(v)
        for i, a in enumerate(self.a_grid):
            for j, w in enumerate(self.w_grid):
                v_new[i, j] = u(w + self.a_grid[i] - self.a_grid) + β * ((1 - α) * v[:, j] + α * d)

        return v_new

    def update(self, v, d):

        # split these into get_next_X methods.
        # plot how they respond to stuff
        # something along the lines of high/low wage and high/low assets.
        # so v would be a 2x2 matrix with decisions that we can predict.
        # make sure max_iter isn't reached (i.e thaget_next_X methods.t the Monte Calro really converges)

        d_new = self.get_next_d(v, d)

        v_new = self.get_next_v(v, d)

        return v_new, d_new


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

        if (i == max_iter):
            raise Exception("iteration limit reached without convergence")

    return v, d


def solve_model_with_savings(mcm, tol=1e-5, max_iter=200):
    """
    Iterates to convergence on the Bellman equations

    * mcm is an instance of McCallModel
    """

    v = np.ones_like((mcm.a_grid, mcm.w_grid))    # Initial guess of v
    d = np.ones_like(mcm.a_grid)                           # Initial guess of d
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


def test():
    mcm1 = BaseMcCall()
    mcm2 = McCallWithSavings(
        a_grid_min=100,
        a_grid_max=100,
        a_grid_size=1
    )
    v1, d1 = solve_model(mcm1, max_iter=200)
    v2, d2 = solve_model_with_savings(mcm2, max_iter=200)
    if d1 != d2:
        raise Exception(f"{d1} != {d2}")
    for j in range(len(mcm1.w_grid)):
        if v1[j] != v2[j]:
            raise Exception(f"{v1[j]} != {v2[j]} in row {j}")
test()

# Here’s a function compute_reservation_wage that takes an instance of McCallModelContinuous and returns the associated reservation wage.
# If v(w)<h for all w, then the function returns np.inf

@njit
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
