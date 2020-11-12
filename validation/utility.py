import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from validation.steady_state import get_steady_states


"""
v and h should both increase in assets and in wages.
for every asset level (row), values of h and v above the reservation wage for that asset level should be the same.
below the reservation wage, h should be higher than v, and also constant on that part of the row.
"""


DIR = '/home/shay/projects/quantecon/results/utility/'


def utility_by_assets():
    w_choice_indices = np.arange(0, 10, 2)
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('assets')
        ax.set_ylabel('utility')
        ax.plot(m.a_grid, h[:, w_grid_index], '-', alpha=0.4, color="C1", label="h(:, {w})".format(w=round(w)))
        ax.plot(m.a_grid, v[:, w_grid_index], '-', alpha=0.4, color="C2", label="v(:, {w})".format(w=round(w)))
        steady_states = get_steady_states(a_opt_employed[:, w_grid_index])
        for steady_state in steady_states:
            plt.axvline(x=steady_state)
        ax.legend(loc='lower right')
        plt.savefig(DIR + 'utility_by_assets_at_{w}_wage.png'.format(w=w))
        plt.close()


def utility_by_wage():
    a_choice_indices = np.arange(0, 15, 5)
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wages')
        ax.set_ylabel('utility')
        ax.plot(m.w_grid, h[a_grid_index, :], '-', alpha=0.4, color="C1", label="h({a}, :)".format(a=round(a)))
        ax.plot(m.w_grid, v[a_grid_index, :], '-', alpha=0.4, color="C2", label="v({a}, :)".format(a=round(a)))
        try:
            reservation_wage = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]
            v = np.where(reservation_wage == m.w_grid)[0][0]
            plt.axvline(x=v)
        except IndexError:
            pass
        ax.legend(loc='lower right')
        plt.savefig(DIR + 'utility_by_wage_at_{a}_assets.png'.format(a=a))
        plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    utility_by_assets()
    utility_by_wage()


if __name__ == '__main__':
    main()
