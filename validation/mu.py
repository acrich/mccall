import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from steady_state import get_steady_state
from lifetime import generate_lifetime


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
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{mu}_mu.npy'.format(mu=mu))
            a_opt_unemployed = np.load('npy/a_opt_unemployed_at_{mu}_mu.npy'.format(mu=mu))
            accept_or_reject = np.load('npy/accept_or_reject_at_{mu}_mu.npy'.format(mu=mu))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{mu}_mu.npy'.format(mu=mu), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{mu}_mu.npy'.format(mu=mu), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{mu}_mu.npy'.format(mu=mu), accept_or_reject)
            np.save('npy/h_at_{mu}_mu.npy'.format(mu=mu), h)
            np.save('npy/v_at_{mu}_mu.npy'.format(mu=mu), v)

        unemployment_spell = []
        for i in range(1000):
            a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
            unemployment_spell.append(m.T - np.sum(employment_spells))
        unemployment_spells[mu_index] = np.mean(np.asarray(unemployment_spell))

    fig, ax = plt.subplots()
    ax.set_xlabel('mean wage μ')
    ax.set_ylabel('mean unemployment spells')
    ax.plot(mu_choices, unemployment_spells, '-', alpha=0.4, color="C9", label="mean unemployment spells")
    plt.savefig(DIR + 'unemployment_spells_by_mu.png')
    plt.close()


def steady_state_by_mu():
    w_choice_indices = np.arange(0, 10, 2)
    mu_choices = np.linspace(0.1, 10, 20)
    steady_states = np.empty((len(w_choice_indices), len(mu_choices)))
    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{mu}_mu.npy'.format(mu=mu))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{mu}_mu.npy'.format(mu=mu), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{mu}_mu.npy'.format(mu=mu), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{mu}_mu.npy'.format(mu=mu), accept_or_reject)
            np.save('npy/h_at_{mu}_mu.npy'.format(mu=mu), h)
            np.save('npy/v_at_{mu}_mu.npy'.format(mu=mu), v)

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            steady_states[w_choice_index, mu_index] = get_steady_state(a_opt_employed[:, w_grid_index])

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('mean wage μ')
        ax.set_ylabel('steady-state assets')
        ax.plot(mu_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C7", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_mu_at_{w}_wage.png'.format(w=round(w)))
        plt.close()


def reservation_wage_by_mu():
    mu_choices = np.linspace(0.1, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    reservation_wages = np.empty((len(a_choice_indices), len(mu_choices)))

    for mu_index, mu in enumerate(mu_choices):
        m = Model(μ=mu)
        try:
            accept_or_reject = np.load('npy/accept_or_reject_at_{mu}_mu.npy'.format(mu=mu))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{mu}_mu.npy'.format(mu=mu), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{mu}_mu.npy'.format(mu=mu), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{mu}_mu.npy'.format(mu=mu), accept_or_reject)
            np.save('npy/h_at_{mu}_mu.npy'.format(mu=mu), h)
            np.save('npy/v_at_{mu}_mu.npy'.format(mu=mu), v)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, mu_index] = m.w_grid[np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('mean wage μ')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=round(a)))
        ax.plot(mu_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C3", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_mu_with_{a}_assets.png'.format(a=round(a)))
        plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    unemployment_spells_by_mu()
    steady_state_by_mu()
    reservation_wage_by_mu()


if __name__ == '__main__':
    main()
