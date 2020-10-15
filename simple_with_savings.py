from interpolation import interp
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt

c = 1e-10
u = lambda x: np.log(x)
β = 0.96
α = 0.1

w_min = 1e-10
w_max = 50
w_size = 100

w_grid = np.linspace(w_min, w_max, w_size)

a_min = 1e-10
a_max = 50
a_grid = np.linspace(a_min, a_max, 200)


def lognormal_draws(n=100, μ=2.5, σ=1.4, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws


w_draws = lognormal_draws(n=1000, μ=0.5, σ=1)


def update(v, h):

    hf = []
    for i, a in enumerate(a_grid):
        hf.append(lambda x: interp(w_grid, h[i], x))

    d = np.empty_like(a_grid)
    for i, a in enumerate(a_grid):
        d[i] = np.mean(hf[i](w_draws))

    v_new = np.empty((len(a_grid), len(w_grid)))
    a_opt_employed = np.empty((len(a_grid), len(w_grid)), dtype=int)
    for i, a in enumerate(a_grid):
        for j, w in enumerate(w_grid):
            rhs = u(w + a_grid[i] - a_grid) + β*((1 - α)*v[:, j] + α*d)
            v_new[i, j] = np.nanmax(rhs)
            a_opt_employed[i, j] = np.nanargmax(rhs)

    h_new = np.empty((len(a_grid), len(w_grid)))
    a_opt_unemployed = np.empty((len(a_grid), len(w_grid)), dtype=int)
    accept_or_reject = np.empty((len(a_grid), len(w_grid)))
    for i, a in enumerate(a_grid):
        for j, w in enumerate(w_grid):
            rhs = u(c + a_grid[i] - a_grid) + β*d
            rhs_opt = np.nanmax(rhs)
            h_new[i, j] = np.maximum(v[i, j], rhs_opt)
            accept_or_reject[i, j] = np.argmax([rhs_opt, v[i, j]])
            if accept_or_reject[i, j] == 0:
                a_opt_unemployed[i, j] = np.nanargmax(rhs)
            else:
                a_opt_unemployed[i, j] = a_opt_employed[i, j]

    return v_new, h_new, accept_or_reject, a_opt_unemployed, a_opt_employed

def solve_model(tol=1e-3, max_iter=2000):
    """
    Iterates to convergence on the Bellman equations
    """

    v = np.ones((len(a_grid), len(w_grid)))    # Initial guess of v
    h = np.ones((len(a_grid), len(w_grid)))    # Initial guess of h
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        v_new, h_new, accept_or_reject, a_opt_unemployed, a_opt_employed = update(v, h)
        error_1 = np.max(np.max(np.abs(v_new - v)))
        error_2 = np.max(np.max(np.abs(h_new - h)))
        error = max(error_1, error_2)
        v = v_new
        h = h_new
        i += 1

        if i == max_iter:
            raise Exception("Reached max_iter without convergence")

    return v, h, accept_or_reject, a_opt_unemployed, a_opt_employed


v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = solve_model()
#print(v)
#print(h)
#print(accept_or_reject)
#print(a_opt_unemployed)
#print(a_opt_employed)
#
# fig, ax = plt.subplots()
# ax.set_xlabel('a_t')
# ax.set_ylabel('a_{t+1}')
#
# ax.plot(a_grid, a_grid, '-', alpha=0.4, label=f"$a$")
# ax.plot(a_grid, a_grid[a_opt_employed[:,40]], '--', alpha=0.4, label=f"$a\_opt\_employed$")
# ax.plot(a_grid, a_grid[a_opt_unemployed[:,40]], '--', alpha=0.4, label=f"$a\_opt\_unemployed$")
# ax.legend(loc='upper left')
# plt.show()

# hf = []
# for i, a in enumerate(a_grid):
#     hf.append(lambda x: interp(w_grid, h[i], x))
#
# d = np.empty_like(a_grid)
# for i, a in enumerate(a_grid):
#     d[i] = np.mean(hf[i](w_draws))
# print(d)


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
a_0 = a_grid[find_nearest_index(a_grid, 5)]
a = np.empty(T + 1)
a[0] = a_0
consumption = np.empty(T)
itu = []
for t in range(T):
    employment_spells[t] = is_employed
    #print("is_employed is {}".format(is_employed))
    #print("wage is {}".format(w[t]))
    w_index = find_nearest_index(w_grid, w_t)
    a_index = np.where(np.isclose(a_grid, a[t]))[0][0]
    #print("wage index on grid is {}".format(w_index))
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
print([(i,v) for i, v in enumerate(u_t)])
print([(i,v) for i, v in enumerate(realized_wage)])
print([(i,v) for i, v in enumerate(employment_spells)])
print([(i,v) for i, v in enumerate(itu)])

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
