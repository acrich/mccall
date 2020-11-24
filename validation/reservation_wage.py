import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

sys.path.append('/home/shay/projects/quantecon')
from separations import binomial_draws
from model import Model
from wage_distribution import lognormal_draws
from agent import generate_lifetime, find_nearest_index


np.set_printoptions(threshold=sys.maxsize)


DIR = '/home/shay/projects/quantecon/results/reservation_wage/'


"""
reservation wage per asset level - rich and poor make different decisions

accept_or_reject determines the reservation wage. for every savings level it gives a threshold of wage above which we move from reject to accept.
that's all the information that's conveyed there. we expect the reservation wage to increase with savings.
the reservation wage will increase in benefits.
it'll increase in beta.
it'll increase with the mean wage. it'll increase with the variation of the wage distribution because more risk is better in this model.
"""


def assets():
    m = Model()
    try:
        accept_or_reject = np.load('npy/accept_or_reject.npy')
    except IOError:
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        np.save('npy/v.npy', v)
        np.save('npy/h.npy', h)
        np.save('npy/accept_or_reject.npy', accept_or_reject)
        np.save('npy/a_opt_unemployed.npy', a_opt_unemployed)
        np.save('npy/a_opt_employed.npy', a_opt_employed)

    reservation_wage = np.empty_like(m.a_grid)
    for i, a in enumerate(m.a_grid):
        reservation_wage[i] = np.argwhere(accept_or_reject[i, :]  == 1)[0][0]

    fig, ax = plt.subplots()
    ax.set_xlabel('assets level')
    ax.set_ylabel('reservation wage')
    ax.plot(m.a_grid, reservation_wage, '-', alpha=0.4, color="C5", label=f"$$")
    plt.savefig(DIR + 'reservation_wage_by_assets.png')
    plt.close()


def beta():
    m = Model()
    num_beta_choices=20
    a_choices = [0, 10, 91]
    a_grid_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    beta_choices = np.linspace(0.1, 0.95, num_beta_choices)
    results = np.empty((len(a_grid_indices), num_beta_choices))
    for choice_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            accept_or_reject = np.load('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)

        for index_in_indices, grid_index in enumerate(a_grid_indices):
            results[index_in_indices, choice_index] = np.argwhere(accept_or_reject[grid_index, :] == 1)[0][0]

    for index_in_indices, grid_index in enumerate(a_grid_indices):
        a = m.a_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel("discount factor β")
        ax.set_ylabel('reservation_wage (with {a} assets)'.format(a=round(a)))
        ax.plot(beta_choices, results[index_in_indices, :], '-', alpha=0.4, color="C2", label=f"$$")
        plt.savefig(DIR + 'reservation_wage_by_β_with_{a}_assets.png'.format(a=round(a)))
        plt.close()


def benefits():
    m = Model()
    num_z_choices=20
    a_choices = [0, 3, 12]
    a_grid_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    z_choices = np.linspace(0, 10, num_z_choices)
    results = np.empty((len(a_grid_indices), num_z_choices))
    for choice_index, z in enumerate(z_choices):
        m = Model(z=z)
        try:
            accept_or_reject = np.load('npy/accept_or_reject_at_{z}_z.npy'.format(z=z))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/accept_or_reject_at_{z}_z.npy'.format(z=z), accept_or_reject)

        for index_in_indices, grid_index in enumerate(a_grid_indices):
            results[index_in_indices, choice_index] = np.argwhere(accept_or_reject[grid_index, :] == 1)[0][0]

    for index_in_indices, grid_index in enumerate(a_grid_indices):
        a = m.a_grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel("z (unemployment benefits)")
        ax.set_ylabel('reservation_wage (with {a} assets)'.format(a=round(a)))
        ax.plot(z_choices, results[index_in_indices, :], '-', alpha=0.4, color="C2", label=f"$$")
        plt.savefig(DIR + 'reservation_wage_by_z_with_{a}_assets.png'.format(a=round(a)))
        plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    assets()
    benefits()
    beta()


if __name__ == '__main__':
    main()
