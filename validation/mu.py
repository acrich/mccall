import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from test_consumption_savings import get_steady_state

"""
changes in mean wage make potential future unemployment less threatening, but increases the benefit from higher assets in the future.
so there are effects on a_opt_employed in both ways and we can't anticipate results.
changes in the mean wage should encourage unemployed to stay that way longer, so it should decrease asset burn rate, meaning a_opt_unemployed will be higher.
"""


DIR = '/home/shay/projects/quantecon/results/mu/'


def unemployment_spells_by_mu():
    mu_choices = np.linspace(0.1, 10, 20)
    unemployment_spells = np.empty((len(mu_choices)))

    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        unemployment_spell = []
        T = 100
        for i in range(1000):
            a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
            unemployment_spell.append(T - np.sum(employment_spells))
        unemployment_spells[mu_index] = np.mean(np.asarray(unemployment_spell))

    fig, ax = plt.subplots()
    ax.set_xlabel('mean wage μ')
    ax.set_ylabel('mean unemployment spells')
    ax.plot(mu_choices, unemployment_spells, '-', alpha=0.4, color="C1", label="mean unemployment spells")
    plt.savefig(DIR + 'unemployment_spells_by_mu.png')
    plt.close()


def steady_state_by_mu():
    w_choice_indices = np.arange(0, 10, 2)
    mu_choices = np.linspace(0.1, 10, 20)
    steady_states = np.empty((len(w_choice_indices), len(mu_choices)))
    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            ss = get_steady_state(a_opt_employed[:, w_grid_index])
            if len(ss) == 0:
                raise Exception("couldn't find steady state for {}".format(j))
            if len(ss) > 1:
                print(ss)
            steady_states[w_choice_index, mu_index] = next(iter(ss))

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('mean wage μ')
        ax.set_ylabel('steady-state assets')
        ax.plot(mu_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C1", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_mu_at_{w}_wage.png'.format(w=w))
        plt.close()


def savings_by_mu():
    """ i don't trust steady-states because there are too many of them. """
    w_choice_indices = np.arange(0, 10, 2)
    mu_choices = np.linspace(0.1, 10, 20)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(mu_choices), m.a_size))
    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, mu_index, :] = a_opt_employed[:, w_grid_index]

    for mu_index, mu in enumerate(mu_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, mu_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'savings_at_{w}_wage_and_{mu}_mu.png'.format(w=w, mu=mu))
            plt.close()


def unsaving_by_mu():
    w_choice_indices = np.arange(0, 10, 2)
    mu_choices = np.linspace(0.1, 10, 20)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(mu_choices), m.a_size))
    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, mu_index, :] = a_opt_unemployed[:, w_grid_index]

    for mu_index, mu in enumerate(mu_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, mu_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'unsavings_at_{w}_wage_and_{mu}_mu.png'.format(w=w, mu=mu))
            plt.close()


def reservation_wage_by_mu():
    mu_choices = np.linspace(0.1, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    reservation_wages = np.empty((len(a_choice_indices), len(mu_choices)))

    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, mu_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('mean wage μ')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(mu_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_mu_with_{a}_assets.png'.format(a=a))
        plt.close()


def h_by_mu():
    mu_choices = np.linspace(0.1, 10, 20)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    h_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(mu_choices)))

    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                h_results[a_choice_index, w_choice_index, mu_index] = h[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('mean wage μ')
            ax.set_ylabel('h (utility in unemployment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(mu_choices, h_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'h_per_mu_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def v_by_mu():
    mu_choices = np.linspace(0.1, 10, 20)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    v_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(mu_choices)))

    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                v_results[a_choice_index, w_choice_index, mu_index] = v[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('mean wage μ')
            ax.set_ylabel('v (utility in employment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(mu_choices, v_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'v_per_mu_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def validate():
    unemployment_spells_by_mu()
    steady_state_by_mu()
    reservation_wage_by_mu()
    h_by_mu()
    v_by_mu()
    savings_by_alpha()
    unsavings_by_alpha()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    validate()
