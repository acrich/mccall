import sys
from interpolation import interp
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt
from numba import njit, prange, float64, int64
import quantecon as qe


np.set_printoptions(threshold=sys.maxsize)


# @njit
# def u(c, σ=3.0):
#     return (c**(1 - σ) - 1) / (1 - σ)

VERY_SMALL_NUMBER = -1e+3


@njit
def u(x):
    return np.log(x)


def lognormal_draws(n=100, μ=2.5, σ=1.4, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z) * 4
    return w_draws


@njit
def update_d(h):
    d = np.empty_like(a_grid)
    for i, a in enumerate(a_grid):
        hf = lambda x: interp(w_grid, h[i], x)
        d[i] = np.mean(hf(w_draws))
    for i in range(len(d)):
        if np.isnan(d[i]):
            raise Exception("d is NaN")
    return d


@njit
def update_v(v, h, d):
    v_new = np.empty((a_size, w_size))
    a_opt_employed = np.empty((a_size, w_size))
    a_opt_employed = a_opt_employed.astype(int64)
    for i, a in enumerate(a_grid):
        for j, w in enumerate(w_grid):
            consumption = w + a_grid[i]*(1 + r) - a_grid - minimal_consumption

            negative_elements = np.where(consumption < 0)[0]
            if len(negative_elements) > 0:
                z = negative_elements[0] # index_of_first_negative_element
            else:
                z = a_size

            if z == 0:
                # all consumption choices are negative.
                v_new[i, j] = np.nan
                a_opt_employed[i, j] = 0
            else:
                rhs = u(consumption[:z]) + β*((1 - α)*v[:, j][:z] + α*d[:z])
                v_new[i, j] = np.nanmax(rhs)
                if np.isnan(v_new[i, j]):
                    a_opt_employed[i, j] = 0
                else:
                    a_opt_employed[i, j] = np.where(rhs == v_new[i, j])[0][0]

    return v_new, a_opt_employed


@njit
def update_h(v, h, d, a_opt_employed):
    h_new = np.empty((a_size, w_size))
    a_opt_unemployed = np.empty((a_size, w_size))
    a_opt_unemployed = a_opt_unemployed.astype(int64)
    accept_or_reject = np.empty((a_size, w_size))

    for i in range(a_size):
        consumption = c + a_grid[i]*(1 + r) - a_grid - minimal_consumption

        negative_elements = np.where(consumption < 0)[0]
        if len(negative_elements) > 0:
            z = negative_elements[0] # index_of_first_negative_element
        else:
            z = a_size

        if z == 0:
            # all consumption choices are negative.
            # print("all consumption choices are negative.")
            unemployment_opt = VERY_SMALL_NUMBER
            a_opt = 0
        else:
            unemployment = u(consumption[:z]) + β*d[:z]
            unemployment_opt = np.max(unemployment)
            a_opt = np.argmax(unemployment)

        for j in range(w_size):
            rhs = np.asarray([unemployment_opt, v[i, j]])
            h_new[i, j] = np.nanmax(rhs)
            accept_or_reject[i, j] = np.where(rhs == h_new[i, j])[0][0]

            if accept_or_reject[i, j] == 0:
                a_opt_unemployed[i, j] = a_opt
            else:
                a_opt_unemployed[i, j] = a_opt_employed[i, j]
    return h_new, a_opt_unemployed, accept_or_reject


def update(v, h):
    #qe.tic()
    d = update_d(h)
    #qe.toc()

    #qe.tic()
    v_new, a_opt_employed = update_v(v, h, d)
    #qe.toc()

    #qe.tic()
    h_new, a_opt_unemployed, accept_or_reject = update_h(v, h, d, a_opt_employed)
    #qe.toc()

    return v_new, h_new, accept_or_reject, a_opt_unemployed, a_opt_employed


def solve_model(tol=1e-3, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations
    """

    v = np.ones((a_size, w_size))    # Initial guess of v
    h = np.ones((a_size, w_size))    # Initial guess of h
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, h_new, accept_or_reject, a_opt_unemployed, a_opt_employed = update(v, h)
        error_1 = np.nanmax(np.nanmax(np.abs(v_new - v)))
        error_2 = np.nanmax(np.nanmax(np.abs(h_new - h)))
        error = max(error_1, error_2)
        v = v_new
        h = h_new
        i += 1

        if i == max_iter:
            raise Exception("Reached max_iter without convergence")

    print(i)
    return v, h, accept_or_reject, a_opt_unemployed, a_opt_employed


# see: https://stackoverflow.com/a/2566508/1408861
def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def draw_lifetime(T=100, a_0_value=1):
    stays_employed = binomial_draws(n=T)
    u_t = np.empty(T)
    is_employed = 1
    w = lognormal_draws(n=T, μ=0.5, σ=2)
    employment_spells = np.empty(T)
    w_t = w[0]
    realized_wage = np.empty(T)
    separations = []
    a_0 = a_grid[find_nearest_index(a_grid, a_0_value)]
    a = np.empty(T + 1)
    a[0] = a_0
    consumption = np.empty(T)
    itu = []
    for t in range(T):
        employment_spells[t] = is_employed
        w_index = find_nearest_index(w_grid, w_t)
        a_index = np.where(np.isclose(a_grid, a[t]))[0][0]

        if is_employed:
            a[t+1] = a_grid[a_opt_employed[a_index, w_index]]
            consumption[t] = w_t + a[t] - a[t+1]
            u_t[t] = u(consumption[t])
            is_employed = stays_employed[t]
            if not is_employed:
                separations.append(t)

        else:
            w_t = w[t]
            w_index = find_nearest_index(w_grid, w_t)
            is_employed = accept_or_reject[a_index, w_index]
            if is_employed:
                a[t+1] = a_grid[a_opt_employed[a_index, w_index]]
                consumption[t] = w_t + a[t] - a[t+1]
                u_t[t] = u(consumption[t])
            else:
                a[t+1] = a_grid[a_opt_unemployed[a_index, w_index]]
                consumption[t] = c + a[t] - a[t+1]
                u_t[t] = u(consumption[t])
        realized_wage[t] = w_t
        realized_w_index = find_nearest_index(w_grid, w_t)
        itu.append((v[a_index, realized_w_index], h[a_index, realized_w_index]))
    a = a[:-1].copy()
    return a, u_t, realized_wage, employment_spells, itu, consumption, separations


def draw(a, u_t, realized_wage, employment_spells, itu, consumption, separations, T=100):
    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.set_ylabel('stuff')

    ax.plot(range(T), u_t, '--', alpha=0.4, label=f"$u(consumption)$")
    ax.plot(range(T), a, '-', alpha=0.4, label=f"$a_t$")
    ax.plot(range(T), consumption, '-', alpha=0.4, label=f"$c_t$")
    ax.plot(range(T), realized_wage, '--', alpha=0.4, label=f"$w_t$")
    ax.plot(range(T), employment_spells, '--', alpha=0.4, label=f"$employed$")
    for t in separations:
        plt.axvline(x=t)
    ax.legend(loc='upper right')
    plt.show()


def run():
    T = 100
    a, u_t, realized_wage, employment_spells, itu, consumption, separations = draw_lifetime(T=T, a_0_value=1)
    #draw(a, u_t, realized_wage, employment_spells, itu, consumption, separations, T=T)
    print(np.sum(employment_spells))
    print(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
    a, u_t, realized_wage, employment_spells, itu, consumption, separations = draw_lifetime(T=T, a_0_value=150)
    print(np.sum(employment_spells))
    print(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
    #draw(a, u_t, realized_wage, employment_spells, itu, consumption, separations, T=T)


c = 1e-10
β = 0.96
α = 0.1

w_draws = lognormal_draws(n=1000, μ=0.8, σ=0.7)

w_min = 1e-10
w_max = np.max(w_draws)
w_size = 50

print("wage grid is from {min} to {max}, with size {size}, and the average wage is {avg}".format(
    min=w_min,
    max=w_max,
    size=w_size,
    avg=np.mean(w_draws)
))
w_grid = np.linspace(w_min, w_max, w_size)

a_min = 1e-10
a_max = 200
a_size = 100
a_grid = np.linspace(a_min, a_max, a_size)

minimal_consumption = 10


ism = 1  # inter-temporal savings motive
r = (ism/β) - 1




# try:
#     v = np.load('v.npy')
#     h = np.load('h.npy')
#     accept_or_reject = np.load('accept_or_reject.npy')
#     a_opt_unemployed = np.load('a_opt_unemployed.npy')
#     a_opt_employed = np.load('a_opt_employed.npy')
# except IOError:
#     v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = solve_model()
#     np.save('v.npy', v)
#     np.save('h.npy', h)
#     np.save('accept_or_reject.npy', accept_or_reject)
#     np.save('a_opt_unemployed.npy', a_opt_unemployed)
#     np.save('a_opt_employed.npy', a_opt_employed)

v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = solve_model()

print(accept_or_reject)

run()
