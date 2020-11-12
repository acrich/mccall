import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from agent import generate_lifetime
from validation.steady_state import get_steady_state


"""
changes in interest make potential future unemployment less threatening, but increases the benefit from higher assets in the future.
so there are effects on a_opt_employed in both ways and we can't anticipate results.
changes in the interest should encourage unemployed to stay that way longer, so it should decrease asset burn rate, meaning a_opt_unemployed will be higher.

higher ism (consequently also interest), means both a_opt choices should be higher, unless you're unemployed and assets are low, in which case you don't care about interest.

the effect of ism on savings should increase in the level of assets, and in wages.
"""


DIR = '/home/shay/projects/quantecon/results/interest/'


def unemployment_spells_by_interest():
    ism_choices = np.linspace(0.5, 1.5, 10)
    m = Model()
    interest_choices = (ism_choices/m.β) - 1
    unemployment_spells = np.empty((len(interest_choices)))
    for ism_index, ism in enumerate(ism_choices):
        m = Model(ism=ism)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        unemployment_spell = []
        T = 100
        for i in range(1000):
            a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
            unemployment_spell.append(T - np.sum(employment_spells))
        unemployment_spells[ism_index] = np.mean(np.asarray(unemployment_spell))

    fig, ax = plt.subplots()
    ax.set_xlabel('interest rate on assets')
    ax.set_ylabel('mean unemployment spells')
    ax.plot(interest_choices, unemployment_spells, '-', alpha=0.4, color="C1", label="mean unemployment spells")
    plt.savefig(DIR + 'unemployment_spells_by_interest.png')
    plt.close()


def steady_state_by_interest():
    w_choice_indices = np.arange(0, 10, 2)
    interest_choices = np.linspace(0.5, 1.5, 10)
    steady_states = np.empty((len(w_choice_indices), len(interest_choices)))
    for interest_index, interest in enumerate(interest_choices):
        m = Model(ism=interest)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            steady_states[w_choice_index, interest_index] = get_steady_state(a_opt_employed[:, w_grid_index])

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('interest rate on assets')
        ax.set_ylabel('steady-state assets')
        ax.plot(interest_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C1", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_interest_at_{w}_wage.png'.format(w=w))
        plt.close()


def savings_by_interest():
    """ i don't trust steady-states because there are too many of them. """
    w_choice_indices = np.arange(0, 10, 2)
    interest_choices = np.linspace(0.5, 1.5, 10)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(interest_choices), m.a_size))
    for interest_index, interest in enumerate(interest_choices):
        m = Model(ism=interest)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, interest_index, :] = a_opt_employed[:, w_grid_index]

    for interest_index, interest in enumerate(interest_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, interest_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'savings_at_{w}_wage_and_{interest}_interest.png'.format(w=w, interest=interest))
            plt.close()


def unsaving_by_interest():
    w_choice_indices = np.arange(0, 10, 2)
    interest_choices = np.linspace(0.5, 1.5, 10)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(interest_choices), m.a_size))
    for interest_index, interest in enumerate(interest_choices):
        m = Model(ism=interest)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, interest_index, :] = a_opt_unemployed[:, w_grid_index]

    for interest_index, interest in enumerate(interest_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, interest_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'unsavings_at_{w}_wage_and_{interest}_interest.png'.format(w=w, interest=interest))
            plt.close()


def reservation_wage_by_interest():
    interest_choices = np.linspace(0.5, 1.5, 10)
    a_choice_indices = np.arange(0, 15, 5)
    reservation_wages = np.empty((len(a_choice_indices), len(interest_choices)))

    for interest_index, interest in enumerate(interest_choices):
        m = Model(ism=interest)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, interest_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('interest rate on assets')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(interest_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_interest_with_{a}_assets.png'.format(a=a))
        plt.close()


def h_by_interest():
    interest_choices = np.linspace(0.5, 1.5, 10)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    h_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(interest_choices)))

    for interest_index, interest in enumerate(interest_choices):
        m = Model(ism=interest)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                h_results[a_choice_index, w_choice_index, interest_index] = h[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('interest rate on assets')
            ax.set_ylabel('h (utility in unemployment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(interest_choices, h_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'h_per_interest_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def v_by_interest():
    interest_choices = np.linspace(0.5, 1.5, 10)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    v_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(interest_choices)))

    for interest_index, interest in enumerate(interest_choices):
        m = Model(ism=interest)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                v_results[a_choice_index, w_choice_index, interest_index] = v[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('interest rate on assets')
            ax.set_ylabel('v (utility in employment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(interest_choices, v_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'v_per_interest_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    unemployment_spells_by_interest()
    steady_state_by_interest()
    reservation_wage_by_interest()
    h_by_interest()
    v_by_interest()
    savings_by_alpha()
    unsavings_by_alpha()


if __name__ == '__main__':
    main()
