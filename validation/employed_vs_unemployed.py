import os
from model_with_risk import Model
import numpy as np
import matplotlib.pyplot as plt

from test_consumption_savings import get_steady_state


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


# ss by wage for employed/unemployed
def ss_by_wage():
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
    steady_states_employed = np.empty_like(m.w_grid)
    steady_states_unemployed = np.empty_like(m.w_grid)
    for j, w in enumerate(m.w_grid):
        try:
            ss_employed = get_steady_state(a_opt_employed[:, j])
        except:
            print("got exception at {w} for employed".format(w=w))
            continue

        if len(ss_employed) == 0:
            raise Exception("couldn't find steady state for {}".format(j))
        if len(ss_employed) > 1:
            print(ss_employed)

        try:
            ss_unemployed = get_steady_state(a_opt_unemployed[:, j])
        except:
            print("got exception at {w} for unemployed".format(w=w))
            continue

        if len(ss_unemployed) == 0:
            raise Exception("couldn't find steady state for {}".format(j))
        if len(ss_unemployed) > 1:
            print(ss_unemployed)

        steady_states_employed[j] = next(iter(ss_employed))
        steady_states_unemployed[j] = next(iter(ss_unemployed))

    fig, ax = plt.subplots()
    ax.set_xlabel('wage')
    ax.set_ylabel('steady-state assets')
    ax.plot(m.w_grid, steady_states_employed, '-', alpha=0.4, color="C1", label="$ss employed$")
    ax.plot(m.w_grid, steady_states_unemployed, '-', alpha=0.4, color="C1", label="$ss unemployed$")
    ax.legend(loc='upper right')
    plt.savefig(DIR + 'steady_state_by_wage_employed_or_not.png')
    plt.close()

# next period assets by current period assets for employed/unemployed, below reservation wage
# next period assets by current period assets for employed/unemployed, at reservation wage
# next period assets by current period assets for employed/unemployed, above reservation wage
def savings_by_assets():
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

    w_choice_indices = np.arange(0, 6, 2)
    for grid_index in w_choice_indices:
        w = m.w_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('current period assets')
        ax.set_ylabel('next period assets with {w} wage'.format(w=w))
        reservation_wage = np.empty_like(m.a_grid)
        for i, a in enumerate(m.a_grid):
            reservation_wage[i] = np.argwhere(accept_or_reject[i, :]  == 1)[0][0]

        ax.plot(m.a_grid, m.a_grid, '-', alpha=0.4, color="C1", label="$a$")
        ax.plot(m.a_grid, a_opt_employed[:, grid_index], '-', alpha=0.4, color="C2", label="$a_e'$")
        ax.plot(m.a_grid, a_opt_unemployed[:, grid_index], '-', alpha=0.4, color="C3", label="$a_u'$")
        ax.plot(m.a_grid, reservation_wage, '-', alpha=0.4, color="C4", label="$\overline{w}$")
        try:
            v = np.where(reservation_wage == w)[0][0]
            plt.axvline(x=v)
        except IndexError:
            pass
        ax.legend(loc='upper right')
        plt.savefig(DIR + 'savings_by_current_assets_at_{w}_wage_employed_vs_unemployed.png'.format(w=w))
        plt.close()


# a_opt_unemployed by wage, given asset level (mark vertical line for reservation wage)
# a_opt_employed by wage, given asset level (mark vertical line for reservation wage)
def savings_by_wage():
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

    a_choice_indices = np.arange(0, 20, 4)
    for grid_index in a_choice_indices:
        a = m.a_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wage')
        ax.set_ylabel('next period assets with {a} assets'.format(a=a))
        reservation_wage = np.argwhere(accept_or_reject[grid_index, :]  == 1)[0][0]

        ax.plot(m.w_grid, a_opt_employed[grid_index, :], '-', alpha=0.4, color="C2", label="$a_e'$")
        ax.plot(m.w_grid, a_opt_unemployed[grid_index, :], '-', alpha=0.4, color="C3", label="$a_u'$")
        try:
            v = np.where(reservation_wage == m.w_grid)[0][0]
            plt.axvline(x=v)
        except IndexError:
            pass
        ax.legend(loc='upper right')
        plt.savefig(DIR + 'savings_by_wage_at_{a}_assets_employed_vs_unemployed.png'.format(a=a))
        plt.close()


def main():
    ss_by_wage()
    savings_by_assets()
    savings_by_wage()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    main()
