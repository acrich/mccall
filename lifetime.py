import sys
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt
from numba import njit
from model import Model
from wage_distribution import lognormal_draws


# force print to display entire matrices
np.set_printoptions(threshold=sys.maxsize)


# see: https://stackoverflow.com/a/2566508/1408861
def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def generate_lifetime(a_0=1, model={}, accept_or_reject=None, a_opt_unemployed=None, a_opt_employed=None):
    """
    Given initial asset level a_0, this function returns employment,
    consumption and savings decisions and separations and wage incidences
    for an agent living model.T periods.
    """

    # T binomial draws that determine for every period t whether a separation occurs
    # we only look at these draws in periods where is_employed=1.
    is_separated_at = binomial_draws(n=model.T, α=model.α)

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
    employment_spells = np.empty(model.T)

    # offered wage at period t (vector of size T).
    # this vector is only used if the agent is currenly unemployed.
    # this is the actual wage and not a grid index, and the wage drawn may not even
    # be on the grid, so we approximate using find_nearest_index wherever necessary.
    # @TODO: drop this. we're using persistence instead.
    offered_wage_at = lognormal_draws(n=model.T, μ=model.μ, σ=model.σ)

    # wage of employed worker, or offered wage for the unemployed.
    # this is a variable and not a vector - it only states the current wage.
    # this is the actual wage and not a grid index.
    # initial wage set so as to cover minimal consumption constraint and not much more.
    w_t = model.w_grid[find_nearest_index(model.w_grid, model.c_hat) + 1]

    # a vector of size T that holds the offered/current wage for every t.
    # these are the actual wage levels and not grid indices.
    # realized wages may not always fit on the grid, as their taken from
    # draws on a lognormal distribution.
    realized_wage = np.empty(model.T)

    # this is the minimal offered wage that will be accepted.
    # it changes with assets. an agent with a high level of assets may choose
    # to eat them away until a higher wage offer arives, while a poorer agent
    # may choose to accept job offers even if the offered wage is low.
    # this is the actual wage and not a grid index, but the wages are always
    # on the grid, meaning that the real reservation wage, if calculated, may be
    # slightly different than the one represented in this vector, but for all
    # computational purposes this shouldn't make a difference.
    reservation_wage = np.empty(model.T)

    # level of assets (savings decision) at period t.
    # this has T+1 values because, although we only draw T periods,
    # there's still a savings decision made at period T for period T+1.
    # the value at t=0 is the initial assets of the agent.
    # these initial assets are the source of wealth heterogeneity in the model.
    # this vector holds the actual asset levels and not their index on the grid.
    a = np.empty(model.T + 1)
    a[0] = model.a_grid[find_nearest_index(model.a_grid, a_0)]

    # a vector of size T, indicating consumption at period t.
    # consumption represents extra consumption above the minimal level requried
    # by every agent at every period. consumption is everything earned or stored
    # as assets that's not saved as next period assets, so the equation is:
    # consumption - c_hat + next_period_assets = wage_or_unemployment_benefits + current_period_assets * (1 + r)
    consumption = np.empty(model.T)

    # vector of size T holding the single period t utility from consumption.
    # this is equivalent to np.log(consumption).
    # consumption may be very small, so the utility may be negative, but consumption
    # should never be negative, so utility should always be defined.
    u_t = np.empty(model.T)

    for t in range(model.T):
        w_index = find_nearest_index(model.w_grid, w_t)
        a_index = find_nearest_index(model.a_grid, a[t])

        # save current employment status
        employment_spells[t] = is_employed

        # calculate reservation wage given current level of assets.
        reservation_wage[t] = model.w_grid[np.where(accept_or_reject[a_index, :] == 1)[0][0]]

        if is_employed:
            # a_opt_employed is a matrix of next-period asset levels given current period assets and wage.
            a[t+1] = model.a_grid[a_opt_employed[a_index, w_index]]
            # this equation comes from the budget constraint. see model.py for details.
            consumption[t] = w_t + a[t]*(1 + model.r) - a[t+1] - model.c_hat
            # an employed worker is separated from her job at probability α, each period
            is_employed = is_separated_at[t]
            if not is_employed:
                separations.append(t)

        else:
            # w_t = offered_wage_at[t]
            w_t = model.get_wage_offer(w_t)

            w_index = find_nearest_index(model.w_grid, w_t)
            # accept_or_reject is a matrix of job taking decisions given current assets and wage.
            # a value of 1 means the agent will accept the job with the currently offered wage.
            is_employed = accept_or_reject[a_index, w_index]
            employment_spells[t] = is_employed

            if is_employed:
                a[t+1] = model.a_grid[a_opt_employed[a_index, w_index]]
                consumption[t] = w_t + a[t]*(1 + model.r) - a[t+1] - model.c_hat
            else:
                a[t+1] = model.a_grid[a_opt_unemployed[a_index, w_index]]
                # this equation is the same as before, only in the case of unemployment,
                # agents receive unemployment benefits (z) instead of wage (w).
                consumption[t] = model.z + a[t]*(1 + model.r) - a[t+1] - model.c_hat

        u_t[t] = model.u(consumption[t])
        realized_wage[t] = w_t

    # drop assets at period T+1 because we only plot up to period T.
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


def plot_lifetime(a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage, T=100):
    """ plots assets, wages, and decisions over an agent's entire life span """
    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.set_ylabel('stuff')

    ax.plot(range(T), a, '-', alpha=0.4, color="C1", label=f"$a_t$")
    ax.plot(range(T), realized_wage, '--', alpha=0.4, color="C3", label=f"$w_t$")
    ax.plot(range(T), employment_spells, '--', alpha=0.4, color="C6", label=f"$employed$")
    ax.plot(range(T), consumption, '--', alpha=0.4, color="C7", label=f"consumption")
    ax.plot(range(T), reservation_wage, '--', alpha=0.4, color="C8", label="$\overline{w}$")
    for t in separations:
        plt.axvline(x=t, color="C7")
    ax.legend(loc='upper right')
    plt.savefig('results/lifetime_with_{a}_assets.png'.format(a=a[0]))
    plt.show()


def get_stats_from_multiple_lifetimes(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    """ generate averages for end-of-life wages and assets for 1,000 rich and poor agents """
    wages = []
    assets = []
    for i in range(1000):
        a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=0, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
        wages.append(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
        assets.append(a[m.T-1])
    print("average lifetime wage for the poor: {}".format(np.mean(np.asarray(wages))))
    print("assets at end of life for the poor: {}".format(np.mean(assets)))
    wages = []
    assets = []
    for i in range(1000):
        a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=150, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
        wages.append(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
        assets.append(a[m.T-1])
    print("average lifetime wage for the rich: {}".format(np.mean(np.asarray(wages))))
    print("assets at end of life for the rich: {}".format(np.mean(assets)))


def plot_single_lifetime(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    """ plot lifetime assets, wage and decisions for one poor and one rich agent """
    a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=0, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
    plot_lifetime(a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage, T=m.T)
    print(np.sum(employment_spells))
    print(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
    a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=1000, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
    print(np.sum(employment_spells))
    print(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
    plot_lifetime(a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage, T=m.T)


def get_last_wage(a_0, model, accept_or_reject, a_opt_unemployed, a_opt_employed):
    a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=0, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
    return realized_wage[-1]


def plot_stationary_distributions(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    """ plot histograms for wages and assets of 4,000 identical agents at end-of-life """
    wages = []
    for i in range(4000):
        print("%d / 4000" % i)
        wages.append(get_last_wage(0, m, accept_or_reject, a_opt_unemployed, a_opt_employed))

    count, bins, ignored = plt.hist(wages, 200, density=True)
    plt.savefig('results/stationary_wage_distribution.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    m = Model()
    # try:
    #     v = np.load('npy/v.npy')
    #     h = np.load('npy/h.npy')
    #     accept_or_reject = np.load('npy/accept_or_reject.npy')
    #     a_opt_unemployed = np.load('npy/a_opt_unemployed.npy')
    #     a_opt_employed = np.load('npy/a_opt_employed.npy')
    # except IOError:
    #     v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
    #     np.save('npy/v.npy', v)
    #     np.save('npy/h.npy', h)
    #     np.save('npy/accept_or_reject.npy', accept_or_reject)
    #     np.save('npy/a_opt_unemployed.npy', a_opt_unemployed)
    #     np.save('npy/a_opt_employed.npy', a_opt_employed)
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

    plot_single_lifetime(m, accept_or_reject, a_opt_unemployed, a_opt_employed)
    #get_stats_from_multiple_lifetimes(m, accept_or_reject, a_opt_unemployed, a_opt_employed)
    #plot_stationary_distributions(m, accept_or_reject, a_opt_unemployed, a_opt_employed)
