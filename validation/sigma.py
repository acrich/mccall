import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from validation.steady_state import get_steady_state

"""
a mean preserving spread in wages would make unemployment more attractive, so savings in both employment and unemployment should be higher.
"""

DIR = '/home/shay/projects/quantecon/results/sigma/'


def unemployment_spells_by_sigma():
    sigma_choices = np.linspace(0.1, 5, 20)
    unemployment_spells = np.empty((len(sigma_choices)))

    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        unemployment_spell = []
        T = 100
        for i in range(1000):
            a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
            unemployment_spell.append(T - np.sum(employment_spells))
        unemployment_spells[sigma_index] = np.mean(np.asarray(unemployment_spell))

    fig, ax = plt.subplots()
    ax.set_xlabel('wage variance σ')
    ax.set_ylabel('mean unemployment spells')
    ax.plot(sigma_choices, unemployment_spells, '-', alpha=0.4, color="C1", label="mean unemployment spells")
    plt.savefig(DIR + 'unemployment_spells_by_sigma.png')
    plt.close()


def steady_state_by_sigma():
    w_choice_indices = np.arange(0, 10, 2)
    sigma_choices = np.linspace(0.1, 5, 20)
    steady_states = np.empty((len(w_choice_indices), len(sigma_choices)))
    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            steady_states[w_choice_index, sigma_index] = get_steady_state(a_opt_employed[:, w_grid_index])

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wage variance σ')
        ax.set_ylabel('steady-state assets')
        ax.plot(sigma_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C1", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_sigma_at_{w}_wage.png'.format(w=w))
        plt.close()


def savings_by_sigma():
    """ i don't trust steady-states because there are too many of them. """
    w_choice_indices = np.arange(0, 10, 2)
    sigma_choices = np.linspace(0.1, 5, 20)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(sigma_choices), m.a_size))
    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, sigma_index, :] = a_opt_employed[:, w_grid_index]

    for sigma_index, sigma in enumerate(sigma_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, sigma_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'savings_at_{w}_wage_and_{sigma}_sigma.png'.format(w=w, sigma=sigma))
            plt.close()


def unsaving_by_sigma():
    w_choice_indices = np.arange(0, 10, 2)
    sigma_choices = np.linspace(0.1, 5, 20)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(sigma_choices), m.a_size))
    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, sigma_index, :] = a_opt_unemployed[:, w_grid_index]

    for sigma_index, sigma in enumerate(sigma_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, sigma_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'unsavings_at_{w}_wage_and_{sigma}_sigma.png'.format(w=w, sigma=sigma))
            plt.close()


def reservation_wage_by_sigma():
    sigma_choices = np.linspace(0.1, 5, 20)
    a_choice_indices = np.arange(0, 15, 5)
    reservation_wages = np.empty((len(a_choice_indices), len(sigma_choices)))

    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, sigma_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wage variance σ')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(sigma_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_sigma_with_{a}_assets.png'.format(a=a))
        plt.close()


def h_by_sigma():
    sigma_choices = np.linspace(0.1, 5, 20)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    h_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(sigma_choices)))

    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                h_results[a_choice_index, w_choice_index, sigma_index] = h[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('wage variance σ')
            ax.set_ylabel('h (utility in unemployment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(sigma_choices, h_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'h_per_sigma_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def v_by_sigma():
    sigma_choices = np.linspace(0.1, 5, 20)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    v_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(sigma_choices)))

    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                v_results[a_choice_index, w_choice_index, sigma_index] = v[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('wage variance σ')
            ax.set_ylabel('v (utility in employment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(sigma_choices, v_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'v_per_sigma_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    unemployment_spells_by_sigma()
    steady_state_by_sigma()
    reservation_wage_by_sigma()
    h_by_sigma()
    v_by_sigma()
    savings_by_alpha()
    unsavings_by_alpha()


if __name__ == '__main__':
    main()
