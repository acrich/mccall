import sys
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt
from numba import njit
from model import Model
from wage_distribution import lognormal_draws


"""
general plan:
We've taken the basic McCall model, as implemented in quantecon.org, and plan
to apply the following steps:
1. add separations (following example in quantecon.org)
2. add continuous distribution of w (again, following quantecon.org)
3. add savings. from now on I'm by myself.
4. add multiple agents with heterogeneity in a_0
5. add w that's resolved competitively using a technology that includes a stochastic TFP?
"""


np.set_printoptions(threshold=sys.maxsize)


# see: https://stackoverflow.com/a/2566508/1408861
def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def generate_lifetime(T=100, a_0=1, model={}, accept_or_reject=None, a_opt_unemployed=None, a_opt_employed=None):
    """
    Given initial asset level a_0, this function returns employment,
    consumption and savings decisions and separations and wage incidences
    for an agent living T periods.
    """

    # T binomial draws that determine for every period t whether a separation occurs
    # we only look at these draws in periods where is_employed=1.
    is_separated_at = binomial_draws(n=T, α=model.α)

    # lists periods in which separations occured.
    # a separation would occur in period t if is_employed=1 and is_separated_at[t] = 1.
    separations = []

    # agents always start employed, and earn their initial wage up to the first separation event.
    # this variable holds the employment status of the agent.
    # this is a variable and not a vector - it only states the current employment status.
    # this variable acts like a boolean, but we use {0,1} instead for convenience (this is what
    # comes out of is_separated_at, and what we store in employment_spells).
    is_employed = 1

    # a verctor of length T that states whether the agent was employed or not at every t.
    # if an agent is employed at period t then employment_spells[t] = 1, otherwise 0.
    # we use {0,1} to be able to sketch these results on a line chart, and also for
    # confortably doing vector multiplication with the wage vector, so as to attain a sum
    # of income in all periods.
    employment_spells = np.empty(T)

    # offered wage at period t (vector of size T).
    # this vector is only used if the agent is currenly unemployed.
    # this is the actual wage and not a grid index, and the wage drawn may not even
    # be on the grid, so we approximate using find_nearest_index wherever necessary.
    offered_wage_at = lognormal_draws(n=T, μ=model.μ, σ=model.σ)

    # wage of employed worker, or offered wage for the unemployed.
    # this is a variable and not a vector - it only states the current wage.
    # this is the actual wage and not a grid index.
    w_t = model.w_grid[0]

    # a vector of size T that holds the offered/current wage for every t.
    # these are the actual wage levels and not grid indices.
    # realized wages may not always fit on the grid, as their taken from
    # draws on a lognormal distribution.
    realized_wage = np.empty(T)

    # this is the minimal offered wage that will be accepted.
    # it changes with assets. an agent with a high level of assets may choose
    # to eat them away until a higher wage offer arives, while a poorer agent
    # may choose to accept job offers even if the offered wage is low.
    # this is the actual wage and not a grid index, but the wages are always
    # on the grid, meaning that the real reservation wage, if calculated, may be
    # slightly different than the one represented in this vector, but for all
    # computational purposes this shouldn't make a difference.
    reservation_wage = np.empty(T)

    # level of assets (savings decision) at period t.
    # this has T+1 values because, although we only draw T periods,
    # there's still a savings decision made at period T for period T+1.
    # the value at t=0 is the initial assets of the agent.
    # these initial assets are the source of wealth heterogeneity in the model.
    # this vector holds the actual asset levels and not their index on the grid.
    a = np.empty(T + 1)
    a[0] = model.a_grid[find_nearest_index(model.a_grid, a_0)]

    # a vector of size T, indicating consumption at period t.
    # consumption represents extra consumption above the minimal level requried
    # by every agent at every period. consumption is everything earned or stored
    # as assets that's not saved as next period assets, so the equation is:
    # consumption - c_hat + next_period_assets = wage_or_unemployment_benefits + current_period_assets * (1 + r)
    consumption = np.empty(T)

    # vector of size T holding the single period t utility from consumption.
    # this is equivalent to np.log(consumption).
    # consumption may be very small, so the utility may be negative, but consumption
    # should never be negative, so utility should always be defined.
    u_t = np.empty(T)

    for t in range(T):
        w_index = find_nearest_index(model.w_grid, w_t)
        a_index = find_nearest_index(model.a_grid, a[t])

        employment_spells[t] = is_employed
        reservation_wage[t] = model.w_grid[np.where(accept_or_reject[a_index, :] == 1)[0][0]]

        if is_employed:
            a[t+1] = model.a_grid[a_opt_employed[a_index, w_index]]
            consumption[t] = w_t + a[t] - a[t+1] - model.c_hat
            is_employed = is_separated_at[t]
            if not is_employed:
                separations.append(t)

        else:
            w_t = offered_wage_at[t]
            w_index = find_nearest_index(model.w_grid, w_t)
            is_employed = accept_or_reject[a_index, w_index]
            if is_employed:
                a[t+1] = model.a_grid[a_opt_employed[a_index, w_index]]
                consumption[t] = w_t + a[t] - a[t+1] - model.c_hat
            else:
                a[t+1] = model.a_grid[a_opt_unemployed[a_index, w_index]]
                consumption[t] = model.z + a[t] - a[t+1] - model.c_hat
        u_t[t] = model.u(consumption[t])
        realized_wage[t] = w_t

    # drop assets at period T+1 because our graphs only go up to T.
    a = a[:-1].copy()

    return (
        a,
        u_t,
        realized_wage,
        employment_spells,
        consumption,
        separations,
        reservation_wage
    )


def draw(a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage, T=100):
    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.set_ylabel('stuff')

    # ax.plot(range(T), u_t, '--', alpha=0.4, color="C0", label=f"$u(c_t)$")
    ax.plot(range(T), a, '-', alpha=0.4, color="C1", label=f"$a_t$")
    # ax.plot(range(T), consumption, '-', alpha=0.4, color="C2", label=f"$c_t$")
    ax.plot(range(T), realized_wage, '--', alpha=0.4, color="C3", label=f"$w_t$")
    ax.plot(range(T), employment_spells, '--', alpha=0.4, color="C6", label=f"$employed$")
    ax.plot(range(T), reservation_wage, '--', alpha=0.4, color="C8", label="$\overline{w}$")
    for t in separations:
        plt.axvline(x=t, color="C7")
    ax.legend(loc='upper right')
    plt.show()


def run_a_lot(T, m):
    wages = []
    assets = []
    for i in range(1000):
        a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=0, model=m)
        wages.append(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
        assets.append(a[T-1])
    print("average lifetime wage for the poor: {}".format(np.mean(np.asarray(wages))))
    print("assets at end of life for the poor: {}".format(np.mean(assets)))
    wages = []
    assets = []
    for i in range(1000):
        a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=150, model=m)
        wages.append(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
        assets.append(a[T-1])
    print("average lifetime wage for the rich: {}".format(np.mean(np.asarray(wages))))
    print("assets at end of life for the rich: {}".format(np.mean(assets)))

def run(T, m):
    a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=0, model=m)
    draw(a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage, T=T)
    print(np.sum(employment_spells))
    print(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
    a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=150, model=m)
    print(np.sum(employment_spells))
    print(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
    draw(a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage, T=T)




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

if __name__ == '__main__':
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

    T = 408
    run(T, m)
