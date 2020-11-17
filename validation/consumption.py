import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
sys.path.append('/home/shay/projects/quantecon')

from model import Model
from separations import binomial_draws
from wage_distribution import lognormal_draws
from agent import generate_lifetime
from steady_state import get_steady_state


np.set_printoptions(threshold=sys.maxsize)


"""
there is a steady-state level of assets.
it increases with the wage as expected.

higher level of assets (a) increases both next period asssets (a') and consumption (c).
the extent to which it does either of the two changes with a and w.
with low assets and high wage, a' will be comparatively big to any other scenario.

for any a higher than the steady-state for a given w, a' will be lower than a.

given fixed w, a' increases in a (it may not be strictly monotone, meaning the gap between a and a' may actually decrease).

given fixed a, we expect a' to increase in w. this is generally the case, but
we don't test for it, because it's not always perfectly that way.
with very low wages, sometimes agents preserve their savings, and for slightly
higher wages, they eat away at some of their savings.

for consumption, the formula is:
    c + \overline{c} + a' = (1 + r)a + w
this means that c should increase in a and in w.
to what extent depends on the choice of a'.

increased savings should make for smoother consumption.
we test this by looking at consumption by assets when the wage is low (zero).
indeed we see that consumption increases with assets, which means that agents
eat away savings, thus smoothing consumption to a certain extent.
"""


DIR = '/home/shay/projects/quantecon/results/savings_and_consumption/'


def is_monotonically_increasing(vector):
    if len(vector) == 0:
        return True

    last_value = vector[0]
    for scalar in vector[1:]:
        if scalar < last_value:
            return False
        last_value = scalar
    return True


def ss_by_wage(m, a_opt_employed):
    steady_states = []
    for j, w in enumerate(m.w_grid):
        steady_states.append(get_steady_state(a_opt_employed[:, j]))

    fig, ax = plt.subplots()
    ax.set_xlabel('wage')
    ax.set_ylabel('steady-state assets')
    ax.plot(m.w_grid, steady_states, '-', alpha=0.4, color="C7", label=f"$steady state assets$")
    plt.savefig(DIR + 'steady_state_by_wage.png')
    plt.close()


def savings_is_increasing_in_current_assets(m, a_opt_employed):
    for j, w in enumerate(m.w_grid):
        assert(is_monotonically_increasing(a_opt_employed[:, j]))

    w_choice_indices = np.arange(0, 20, 2)
    for grid_index in w_choice_indices:
        w = m.w_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('current period assets')
        ax.set_ylabel('next period assets with {w} wage'.format(w=round(w)))

        ax.plot(m.a_grid, m.a_grid, '-', alpha=0.4, color="C1", label=f"$a$")
        ax.plot(m.a_grid, a_opt_employed[:, grid_index], '-', alpha=0.4, color="C2", label=f"$a'$")
        ax.legend(loc='upper right')
        plt.savefig(DIR + 'savings_by_current_assets_at_{w}_wage.png'.format(w=round(w)))
        plt.close()


def consumption_by_assets(m, a_opt_employed):
    w_index_choices = np.arange(0, 20)
    for grid_index in w_index_choices:
        w = m.w_grid[grid_index]
        consumption = []
        for i, a in enumerate(m.a_grid):
            c = (1 + m.r)*a + w - m.c_hat - m.a_grid[a_opt_employed[i, grid_index]]
            consumption.append(c)

        fig, ax = plt.subplots()
        ax.set_xlabel('current period assets')
        ax.set_ylabel('consumption given {w} wage'.format(w=round(w)))

        ax.plot(m.a_grid, consumption, '-', alpha=0.4, color="C2", label=f"$c$")
        plt.savefig(DIR + 'consumption_by_current_assets_at_{w}_wage.png'.format(w=round(w)))
        plt.close()


def consumption_by_wage(m, a_opt_employed):
    a_index_choices = np.arange(0, 20)
    for grid_index in a_index_choices:
        a = m.a_grid[grid_index]
        consumption = []
        for j, w in enumerate(m.w_grid):
            c = (1 + m.r)*a + w - m.c_hat - m.a_grid[a_opt_employed[grid_index, j]]
            consumption.append(c)

        fig, ax = plt.subplots()
        ax.set_xlabel('wage')
        ax.set_ylabel('consumption given {a} assets'.format(a=round(a)))

        ax.plot(m.w_grid, consumption, '-', alpha=0.4, color="C2", label=f"$c$")
        plt.savefig(DIR + 'consumption_by_wage_at_{a}_assets.png'.format(a=round(a)))
        plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    m = Model()
    try:
        v = np.load('npy/v.npy')
        h = np.load('npy/h.npy')
        accept_or_reject = np.load('npy/accept_or_reject.npy')
        a_opt_unemployed = np.load('npy/a_opt_unemployed.npy')
        a_opt_employed = np.load('npy/a_opt_employed.npy')
    except IOError:
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        np.save('npy/v.npy', v)
        np.save('npy/h.npy', h)
        np.save('npy/accept_or_reject.npy', accept_or_reject)
        np.save('npy/a_opt_unemployed.npy', a_opt_unemployed)
        np.save('npy/a_opt_employed.npy', a_opt_employed)

    ss_by_wage(m, a_opt_employed)
    savings_is_increasing_in_current_assets(m, a_opt_employed)
    consumption_by_assets(m, a_opt_employed)
    consumption_by_wage(m, a_opt_employed)


if __name__ == '__main__':
    main()
