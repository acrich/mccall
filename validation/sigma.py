import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from steady_state import get_steady_state
from lifetime import generate_lifetime

"""
a mean preserving spread in wages would make unemployment more attractive, so savings in both employment and unemployment should be higher.
"""

DIR = '/home/shay/projects/quantecon/results/sigma/'


def unemployment_spells_by_sigma():
    sigma_choices = np.linspace(0.1, 5, 20)
    unemployment_spells = np.empty((len(sigma_choices)))

    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{sigma}_sigma.npy'.format(sigma=sigma))
            a_opt_unemployed = np.load('npy/a_opt_unemployed_at_{sigma}_sigma.npy'.format(sigma=sigma))
            accept_or_reject = np.load('npy/accept_or_reject_at_{sigma}_sigma.npy'.format(sigma=sigma))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{sigma}_sigma.npy'.format(sigma=sigma), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{sigma}_sigma.npy'.format(sigma=sigma), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{sigma}_sigma.npy'.format(sigma=sigma), accept_or_reject)
            np.save('npy/h_at_{sigma}_sigma.npy'.format(sigma=sigma), h)
            np.save('npy/v_at_{sigma}_sigma.npy'.format(sigma=sigma), v)

        unemployment_spell = []
        for i in range(1000):
            a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
            unemployment_spell.append(m.T - np.sum(employment_spells))
        unemployment_spells[sigma_index] = np.mean(np.asarray(unemployment_spell))

    fig, ax = plt.subplots()
    ax.set_xlabel('wage variance σ')
    ax.set_ylabel('mean unemployment spells')
    ax.plot(sigma_choices, unemployment_spells, '-', alpha=0.4, color="C9", label="mean unemployment spells")
    plt.savefig(DIR + 'unemployment_spells_by_sigma.png')
    plt.close()


def steady_state_by_sigma():
    w_choice_indices = np.arange(0, 10, 2)
    sigma_choices = np.linspace(0.1, 1.5, 10)
    steady_states = np.empty((len(w_choice_indices), len(sigma_choices)))
    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{sigma}_sigma.npy'.format(sigma=sigma))
            a_opt_unemployed = np.load('npy/a_opt_unemployed_at_{sigma}_sigma.npy'.format(sigma=sigma))
            accept_or_reject = np.load('npy/accept_or_reject_at_{sigma}_sigma.npy'.format(sigma=sigma))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{sigma}_sigma.npy'.format(sigma=sigma), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{sigma}_sigma.npy'.format(sigma=sigma), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{sigma}_sigma.npy'.format(sigma=sigma), accept_or_reject)
            np.save('npy/h_at_{sigma}_sigma.npy'.format(sigma=sigma), h)
            np.save('npy/v_at_{sigma}_sigma.npy'.format(sigma=sigma), v)

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            steady_states[w_choice_index, sigma_index] = get_steady_state(a_opt_employed[:, w_grid_index])

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wage variance σ')
        ax.set_ylabel('steady-state assets')
        ax.plot(sigma_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C7", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_sigma_at_{w}_wage.png'.format(w=round(w)))
        plt.close()


def reservation_wage_by_sigma():
    sigma_choices = np.linspace(0.1, 1.5, 10)
    a_choice_indices = np.arange(0, 30, 10)
    reservation_wages = np.empty((len(a_choice_indices), len(sigma_choices)))

    for sigma_index, sigma in enumerate(sigma_choices):
        m = Model(σ=sigma)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{sigma}_sigma.npy'.format(sigma=sigma))
            a_opt_unemployed = np.load('npy/a_opt_unemployed_at_{sigma}_sigma.npy'.format(sigma=sigma))
            accept_or_reject = np.load('npy/accept_or_reject_at_{sigma}_sigma.npy'.format(sigma=sigma))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{sigma}_sigma.npy'.format(sigma=sigma), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{sigma}_sigma.npy'.format(sigma=sigma), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{sigma}_sigma.npy'.format(sigma=sigma), accept_or_reject)
            np.save('npy/h_at_{sigma}_sigma.npy'.format(sigma=sigma), h)
            np.save('npy/v_at_{sigma}_sigma.npy'.format(sigma=sigma), v)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, sigma_index] = m.w_grid[np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('wage variance σ')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=round(a)))
        ax.plot(sigma_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C3", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_sigma_with_{a}_assets.png'.format(a=round(a)))
        plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    unemployment_spells_by_sigma()
    steady_state_by_sigma()
    reservation_wage_by_sigma()


if __name__ == '__main__':
    main()
