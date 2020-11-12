import os
import sys
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt
from numba import njit
from model import Model
from wage_distribution import lognormal_draws
from agent import generate_lifetime


np.set_printoptions(threshold=sys.maxsize)


"""
the lower ism is, the lower the savings decisions should be.
the higher the probability of separations (alpha), the lower the relationship between ism and savings should be, because there's another savings motive.
"""


DIR = '/home/shay/projects/quantecon/results/savings_motives/'


def reservation_wage():
    #ism should interact with ARA. if ism is higher (interest is higher), we'll take more risks in order to increase savings.
    # so the reservation wage should increase with ism, and higher ARA should make the effect smaller.
    ism_choices = np.linspace(0.5, 1.5, 10)
    a_choice_indices = np.arange(0, 30, 3)

    reservation_wages = np.empty((len(a_choice_indices), len(ism_choices)))
    for ism_index, ism in enumerate(ism_choices):
        m = Model(ism=ism)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, ism_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('inter-temporal savings motive')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(ism_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_ism_with_{a}_assets.png'.format(a=a))
        plt.close()


def savings_per_ism():
    a_choice_indices = np.arange(0, 30, 3)
    w_choice_indices = np.arange(0, 10)
    ism_choices = np.linspace(0.5, 1.5, 10)
    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(ism_choices)))
    for ism_index, ism in enumerate(ism_choices):
        m = Model(ism=ism)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, ism_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('inter-temporal savings motive')
            ax.set_ylabel('next period assets for agent with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(ism_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'savings_per_ism_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def savings_per_alpha():
    a_choice_indices = np.arange(0, 30, 3)
    w_choice_indices = np.arange(0, 10)
    alpha_choices = np.linspace(0, 1, 20)
    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(alpha_choices)))
    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, alpha_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('separations rate (alpha)')
            ax.set_ylabel('next period assets for agent with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(alpha_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'savings_per_alpha_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def main():
    test_reservation_wage()
    savings_per_ism()
    savings_per_alpha()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    main()
