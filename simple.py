from interpolation import interp
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt

"""
this is a simplified version of the model - it doesn't have savings.
I only use this for graphing out how things look like initially.
"""

c = 0.5
u = lambda x: np.log(x)
β = 0.96
α = 0.1

w_min = 1e-10
w_max = 10

w_grid = np.linspace(w_min, w_max, 100)


def lognormal_draws(n=100, μ=2.5, σ=0.5, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws

w_draws = lognormal_draws(n=1000, μ=0.5, σ=1)


def update(v, h):

    hf = lambda x: interp(w_grid, h, x)

    d = np.mean(hf(w_draws))

    v_new = np.empty_like(w_grid)
    for i, w in enumerate(w_grid):
        v_new[i] = u(w) + β*((1 - α)*v[i] + α*d)

    h_new = np.empty_like(w_grid)
    accept_or_reject = np.empty_like(w_grid)
    for i, w in enumerate(w_grid):
        h_new[i] = np.maximum(v[i], u(c) + β*d)
        accept_or_reject[i] = np.argmax([u(c) + β*d, v[i]])

    return v_new, h_new, accept_or_reject

def solve_model(tol=1e-3, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations
    """

    v = np.ones_like(w_grid)    # Initial guess of v
    h = np.ones_like(w_grid)    # Initial guess of h
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, h_new, accept_or_reject = update(v, h)
        error_1 = np.max(np.abs(v_new - v))
        error_2 = np.max(np.abs(h_new - h))
        error = max(error_1, error_2)
        v = v_new
        h = h_new
        i += 1

        if i == max_iter:
            raise Exception("Reached max_iter without convergence")

    return v, h, accept_or_reject


v, h, accept_or_reject = solve_model()
print(v)
print(h)
print(accept_or_reject)

reservation_wage = w_grid[np.argmax(accept_or_reject >= 1)]

hf = lambda x: interp(w_grid, h, x)

d = np.mean(hf(w_draws))
print(d)


# see: https://stackoverflow.com/a/2566508/1408861
def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


T = 50
stays_employed = binomial_draws()
u_t = np.empty(T)
is_employed = 1
w = lognormal_draws(n=T, μ=0.5, σ=1)
employment_spells = np.empty(T)
w_t = w[0]
realized_wage = np.empty(T)
separations = []
for t in range(T):
    employment_spells[t] = is_employed
    #print("is_employed is {}".format(is_employed))
    #print("wage is {}".format(w[t]))
    w_index = find_nearest_index(w_grid, w_t)
    #print("wage index on grid is {}".format(w_index))
    if is_employed:
        u_t[t] = u(w_t)
        is_employed = stays_employed[t]
        if not is_employed:
            separations.append(t)
    else:
        w_t = w[t]
        w_index = find_nearest_index(w_grid, w_t)
        is_employed = accept_or_reject[w_index]
        if is_employed:
            u_t[t] = u(w_t)
        else:
            u_t[t] = u(c)
    realized_wage[t] = w_t

print(u_t)
print(realized_wage)
print(employment_spells)

fig, ax = plt.subplots()
ax.set_xlabel('periods')
ax.set_ylabel('stuff')

ax.plot(range(T), u_t, '--', alpha=0.4, label=f"$u(consumption)$")
ax.plot(range(T), realized_wage, '--', alpha=0.4, label=f"$w_t$")
ax.plot(range(T), employment_spells, '--', alpha=0.4, label=f"$employed$")
for t in separations:
    plt.axvline(x=t)
plt.axhline(y=reservation_wage)
ax.legend(loc='upper left')
plt.show()
