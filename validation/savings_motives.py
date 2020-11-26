import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

sys.path.append('/home/shay/projects/quantecon')
from separations import binomial_draws
from model import Model
from wage_distribution import lognormal_draws
from lifetime import generate_lifetime, find_nearest_index


np.set_printoptions(threshold=sys.maxsize)


"""
the lower ism is, the lower the savings decisions should be.
the higher the probability of separations (alpha), the lower the relationship between ism and savings should be, because there's another savings motive.
"""


DIR = '/home/shay/projects/quantecon/results/savings_motives/'


def reservation_wage():
    #ism should interact with ARA. if ism is higher (interest is higher), we'll take more risks in order to increase savings.
    # so the reservation wage should increase with ism, and higher ARA should make the effect smaller.
    m = Model()
    ism_choices = np.linspace(0.5, 1.5, 10)
    interest_choices = (ism_choices/m.β) - 1
    a_choices = [0, 18, 24]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])

    reservation_wages = np.empty((len(a_choice_indices), len(ism_choices)))
    for ism_index, ism in enumerate(ism_choices):
        m = Model(ism=ism)
        try:
            accept_or_reject = np.load('npy/accept_or_reject_at_{ism}_ism.npy'.format(ism=ism))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{ism}_ism.npy'.format(ism=ism), a_opt_employed)
            np.save('npy/accept_or_reject_at_{ism}_ism.npy'.format(ism=ism), accept_or_reject)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, ism_index] = m.w_grid[np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('interest rate r')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=round(a)))
        ax.plot(interest_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C3", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_ism_with_{a}_assets.png'.format(a=round(a)))
        plt.close()


def savings_per_ism():
    m = Model()
    a_choices = [0, 9, 21]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    w_choices = [3]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    ism_choices = np.linspace(0.5, 1.5, 10)
    interest_choices = (ism_choices/m.β) - 1
    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(ism_choices)))
    for ism_index, ism in enumerate(ism_choices):
        m = Model(ism=ism)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{ism}_ism.npy'.format(ism=ism))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{ism}_ism.npy'.format(ism=ism), a_opt_employed)
            np.save('npy/accept_or_reject_at_{ism}_ism.npy'.format(ism=ism), accept_or_reject)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, ism_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('interest rate r')
            ax.set_ylabel('next period assets for agent with {a} assets and {w} wage'.format(a=round(a), w=round(w)))
            ax.plot(interest_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'savings_per_ism_with_{w}_wage_and_{a}_assets.png'.format(a=round(a), w=round(w)))
            plt.close()


def savings_per_alpha():
    m = Model()
    a_choices = [0, 9, 21, 24, 27]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    w_choices = [0, 3, 5]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    alpha_choices = np.linspace(0.05, 0.95, 18)
    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(alpha_choices)))
    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{alpha}_alpha.npy'.format(alpha=alpha))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{alpha}_alpha.npy'.format(alpha=alpha), a_opt_employed)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, alpha_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('separations rate (alpha)')
            ax.set_ylabel('next period assets for agent with {a} assets and {w} wage'.format(a=round(a), w=round(w)))
            ax.plot(alpha_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'savings_per_alpha_with_{w}_wage_and_{a}_assets.png'.format(a=round(a), w=round(w)))
            plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    reservation_wage()
    savings_per_ism()
    savings_per_alpha()


if __name__ == '__main__':
    main()
