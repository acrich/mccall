import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from validation.steady_state import get_steady_state


"""
higher alpha means times of employment I'd want to save more, so a_opt_employed should have a higher equilibrium. a_opt_unemployed should not be affected.
"""


DIR = '/home/shay/projects/quantecon/results/alpha/'


def steady_state_by_alpha():
    """
    plot steady state assets level for employed agents by different values of
    the separation rate (α).
    there is a (non-unique) steady-state level for every wage. so this function
    generates plots for several wage levels. if there are more than 1 steady
    state, we only plot the first one in the list.
    """
    w_choice_indices = np.arange(0, 10, 2)
    alpha_choices = np.linspace(0.05, 0.95, 18)
    steady_states = np.empty((len(w_choice_indices), len(alpha_choices)))
    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            steady_states[w_choice_index, alpha_index] = get_steady_state(a_opt_employed[:, w_grid_index])

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('separations rate α')
        ax.set_ylabel('steady-state assets')
        ax.plot(alpha_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C1", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_alpha_at_{w}_wage.png'.format(w=w))
        plt.close()


def savings_by_alpha():
    """ i don't trust steady-states because there are too many of them. """
    w_choice_indices = np.arange(0, 10, 2)
    alpha_choices = np.linspace(0.05, 0.95, 18)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(alpha_choices), m.a_size))
    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, alpha_index, :] = a_opt_employed[:, w_grid_index]

    for alpha_index, alpha in enumerate(alpha_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, alpha_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'savings_at_{w}_wage_and_{alpha}_alpha.png'.format(w=w, alpha=alpha))
            plt.close()


def unsaving_by_alpha():
    w_choice_indices = np.arange(0, 10, 2)
    alpha_choices = np.linspace(0.05, 0.95, 18)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(alpha_choices), m.a_size))
    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, alpha_index, :] = a_opt_unemployed[:, w_grid_index]

    for alpha_index, alpha in enumerate(alpha_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, alpha_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'unsavings_at_{w}_wage_and_{alpha}_alpha.png'.format(w=w, alpha=alpha))
            plt.close()


def reservation_wage_by_alpha():
    alpha_choices = np.linspace(0.05, 0.95, 18)
    a_choice_indices = np.arange(0, 15, 5)
    reservation_wages = np.empty((len(a_choice_indices), len(alpha_choices)))

    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, alpha_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('separations rate α')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(alpha_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_alpha_with_{a}_assets.png'.format(a=a))
        plt.close()


def h_by_alpha():
    alpha_choices = np.linspace(0.05, 0.95, 18)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    h_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(alpha_choices)))

    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                h_results[a_choice_index, w_choice_index, alpha_index] = h[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('separations rate α')
            ax.set_ylabel('h (utility in unemployment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(alpha_choices, h_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'h_per_alpha_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def v_by_alpha():
    alpha_choices = np.linspace(0.05, 0.95, 18)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    v_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(alpha_choices)))

    for alpha_index, alpha in enumerate(alpha_choices):
        m = Model(α=alpha)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                v_results[a_choice_index, w_choice_index, alpha_index] = v[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('separations rate α')
            ax.set_ylabel('v (utility in employment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(alpha_choices, v_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'v_per_alpha_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    steady_state_by_alpha()
    reservation_wage_by_alpha()
    h_by_alpha()
    v_by_alpha()
    savings_by_alpha()
    unsaving_by_alpha()


if __name__ == '__main__':
    main()
