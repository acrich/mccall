import os
from model import Model
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from steady_state import get_steady_state, get_steady_states
from lifetime import find_nearest_index


"""
a_opt_unemployed should match a_opt_employed for wages above the reservation wage.
for wages below that, a_opt_unemployed should be constant in wage, and increasing in assets
the increase in next period assets is decreasing - at low levels, we eat a lot of our savings. at high levels, we keep relatively most of it, but still eat away absolutely more.
with respect to current level assets, when unemployed, next level assets should always be lower (unless benefits are too high), so we're always below y=x.
when employed, a_opt_employed is higher in wage (and increasing), and higher in assets (but decreasing)
there is an equilibrium asset level for every wage, and households will only save up to it, or eat away savings until they reach it.
so next period assets are above y=x for small current period levels, and eventually they go below.
a_opt_employed shouldn't be directly affected by benefits.
"""


DIR = '/home/shay/projects/quantecon/results/employed_vs_unemployed/'
GRID_LIMIT = 50


# ss by wage for employed/unemployed
def ss_by_wage(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    steady_states_employed = np.empty_like(m.w_grid)
    steady_states_unemployed = np.empty_like(m.w_grid)
    for j, w in enumerate(m.w_grid):
        steady_states_employed[j] = get_steady_state(a_opt_employed[:, j])
        steady_states_unemployed[j] = get_steady_state(a_opt_unemployed[:, j])

    fig, ax = plt.subplots()
    ax.set_xlabel('wage')
    ax.set_ylabel('steady-state assets')
    ax.plot(m.w_grid, steady_states_employed, '-', alpha=0.4, color="C7", label="employed")
    ax.plot(m.w_grid, steady_states_unemployed, '-', alpha=0.4, color="C8", label="unemployed")
    ax.legend(loc='lower right')
    plt.savefig(DIR + 'steady_state_by_wage_employed_or_not.png')
    plt.close()


def steady_states(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    w_choice_indices = np.array([0])
    for w_grid_index in w_choice_indices:
        w = m.w_grid[w_grid_index]
        steady_states = get_steady_states(a_opt_employed[:, w_grid_index])

        fig, ax = plt.subplots()
        ax.set_xlabel('current period assets')
        ax.set_ylabel('next period assets with {w} wage'.format(w=round(w)))

        ax.plot(range(GRID_LIMIT), range(GRID_LIMIT), '-', alpha=0.4, color="C1", label="$a$")
        ax.plot(range(GRID_LIMIT), a_opt_employed[0:GRID_LIMIT, w_grid_index], '-', alpha=0.4, color="C2", label="$a_e'$")
        for steady_state in steady_states:
            plt.axvline(x=steady_state)
        plt.savefig(DIR + 'steady_states_at_{w}_wage.png'.format(w=round(w)))
        plt.close()


# next period assets by current period assets for employed/unemployed, below reservation wage
# next period assets by current period assets for employed/unemployed, at reservation wage
# next period assets by current period assets for employed/unemployed, above reservation wage
def savings_by_assets(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    m = Model()
    w_choices = [0, 7, 20]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    for grid_index in w_choice_indices:
        w = m.w_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('current period assets')
        ax.set_ylabel('next period assets with {w} wage'.format(w=round(w)))
        reservation_wage = np.empty_like(m.a_grid)
        for i, a in enumerate(m.a_grid):
            reservation_wage[i] = np.argwhere(accept_or_reject[i, :]  == 1)[0][0]

        ax.plot(range(GRID_LIMIT), range(GRID_LIMIT), '-', alpha=0.4, color="C1", label="$a$")
        ax.plot(range(GRID_LIMIT), a_opt_employed[0:GRID_LIMIT, grid_index], '-', alpha=0.4, color="C2", label="$a_e'$")
        ax.plot(range(GRID_LIMIT), a_opt_unemployed[0:GRID_LIMIT, grid_index], '-', alpha=0.4, color="C4", label="$a_u'$")
        try:
            v = np.where(reservation_wage == grid_index)[0][0]
            plt.axvline(x=v)
        except IndexError:
            pass
        ax.legend(loc='lower right')
        plt.savefig(DIR + 'savings_by_current_assets_at_{w}_wage_employed_vs_unemployed.png'.format(w=round(w)))
        plt.close()


# a_opt_unemployed by wage, given asset level (mark vertical line for reservation wage)
# a_opt_employed by wage, given asset level (mark vertical line for reservation wage)
def savings_by_wage(m, accept_or_reject, a_opt_unemployed, a_opt_employed):
    m = Model()
    a_choices = [4, 16]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    for grid_index in a_choice_indices:
        a = m.a_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wage')
        ax.set_ylabel('next period assets with {a} assets'.format(a=round(a)))
        reservation_wage = np.argwhere(accept_or_reject[grid_index, :]  == 1)[0][0]

        ax.plot(m.w_grid, a_opt_employed[grid_index, :], '-', alpha=0.4, color="C2", label="$a_e'$")
        ax.plot(m.w_grid, a_opt_unemployed[grid_index, :], '-', alpha=0.4, color="C3", label="$a_u'$")
        try:
            v = np.where(reservation_wage == m.w_grid)[0][0]
            plt.axvline(x=v)
        except IndexError:
            pass
        ax.legend(loc='lower right')
        plt.savefig(DIR + 'savings_by_wage_at_{a}_assets_employed_vs_unemployed.png'.format(a=round(a)))
        plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    m = Model()

    try:
        accept_or_reject = np.load('npy/accept_or_reject.npy')
        a_opt_unemployed = np.load('npy/a_opt_unemployed.npy')
        a_opt_employed = np.load('npy/a_opt_employed.npy')
    except IOError:
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        np.save('npy/accept_or_reject.npy', accept_or_reject)
        np.save('npy/a_opt_unemployed.npy', a_opt_unemployed)
        np.save('npy/a_opt_employed.npy', a_opt_employed)

    ss_by_wage(m, accept_or_reject, a_opt_unemployed, a_opt_employed)
    steady_states(m, accept_or_reject, a_opt_unemployed, a_opt_employed)
    savings_by_assets(m, accept_or_reject, a_opt_unemployed, a_opt_employed)
    savings_by_wage(m, accept_or_reject, a_opt_unemployed, a_opt_employed)


if __name__ == '__main__':
    main()
