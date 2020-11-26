import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from steady_state import get_steady_states
from lifetime import find_nearest_index


"""
v and h should both increase in assets and in wages.
for every asset level (row), values of h and v above the reservation wage for that asset level should be the same.
below the reservation wage, h should be higher than v, and also constant on that part of the row.
"""


DIR = '/home/shay/projects/quantecon/results/utility/'


def utility_by_assets(m, v, h, a_opt_employed):
    m = Model()
    w_choices = [3, 5]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])

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
        plt.savefig(DIR + 'utility_by_assets_at_{w}_wage.png'.format(w=round(w)))
        plt.close()


def utility_by_wage(m, v, h, accept_or_reject):
    m = Model()
    a_choices = [0, 10]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wages')
        ax.set_ylabel('utility')
        ax.plot(m.w_grid, h[a_grid_index, :], '-', alpha=0.4, color="C1", label="h({a}, :)".format(a=round(a)))
        ax.plot(m.w_grid, v[a_grid_index, :], '-', alpha=0.4, color="C2", label="v({a}, :)".format(a=round(a)))
        try:
            reservation_wage = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]
            t = np.where(reservation_wage == range(m.w_size))[0][0]
            plt.axvline(x=t)
        except IndexError:
            pass
        ax.legend(loc='lower right')
        plt.savefig(DIR + 'utility_by_wage_at_{a}_assets.png'.format(a=round(a)))
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

    utility_by_assets(m, v, h, a_opt_employed)
    utility_by_wage(m, v, h, accept_or_reject)


if __name__ == '__main__':
    main()
