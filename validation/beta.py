import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from test_consumption_savings import get_steady_state

"""
higher beta means a_opt_employed and a_opt_unemployed should both be higher.
"""


DIR = '/home/shay/projects/quantecon/results/beta/'


def steady_state_by_beta():
    w_choice_indices = np.arange(0, 10, 2)
    beta_choices = np.linspace(0.5, 0.995, 10)
    steady_states = np.empty((len(w_choice_indices), len(beta_choices)))
    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            ss = get_steady_state(a_opt_employed[:, w_grid_index])
            if len(ss) == 0:
                raise Exception("couldn't find steady state for {}".format(j))
            if len(ss) > 1:
                print(ss)
            steady_states[w_choice_index, beta_index] = next(iter(ss))

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('discount factor β')
        ax.set_ylabel('steady-state assets')
        ax.plot(beta_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C1", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_beta_at_{w}_wage.png'.format(w=w))
        plt.close()



def savings_by_beta():
    """ i don't trust steady-states because there are too many of them. """
    w_choice_indices = np.arange(0, 10, 2)
    beta_choices = np.linspace(0.05, 0.95, 18)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(beta_choices), m.a_size))
    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, beta_index, :] = a_opt_employed[:, w_grid_index]

    for beta_index, beta in enumerate(beta_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, beta_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'savings_at_{w}_wage_and_{beta}_beta.png'.format(w=w, beta=beta))
            plt.close()


def unsaving_by_beta():
    w_choice_indices = np.arange(0, 10, 2)
    beta_choices = np.linspace(0.05, 0.95, 18)
    m = Model()
    savings = np.empty((len(w_choice_indices), len(beta_choices), m.a_size))
    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, beta_index, :] = a_opt_unemployed[:, w_grid_index]

    for beta_index, beta in enumerate(beta_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, savings[w_choice_index, beta_index], '-', alpha=0.4, color="C1", label="next period assets")
            plt.savefig(DIR + 'unsavings_at_{w}_wage_and_{beta}_alpha.png'.format(w=w, alpha=alpha))
            plt.close()


def reservation_wage_by_beta():
    beta_choices = np.linspace(0.5, 0.995, 10)
    a_choice_indices = np.arange(0, 15, 5)
    reservation_wages = np.empty((len(a_choice_indices), len(beta_choices)))

    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, beta_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('discount factor β')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(beta_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_beta_with_{a}_assets.png'.format(a=a))
        plt.close()


def h_by_beta():
    beta_choices = np.linspace(0.5, 0.995, 10)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    h_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(beta_choices)))

    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                h_results[a_choice_index, w_choice_index, beta_index] = h[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('discount factor β')
            ax.set_ylabel('h (utility in unemployment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(beta_choices, h_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'h_per_beta_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def v_by_beta():
    beta_choices = np.linspace(0.5, 0.995, 10)
    w_choice_indices = np.arange(0, 10, 2)
    a_choice_indices = np.arange(0, 15, 5)
    v_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(beta_choices)))

    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                v_results[a_choice_index, w_choice_index, beta_index] = v[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('discount factor β')
            ax.set_ylabel('v (utility in employment) with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(beta_choices, v_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'v_per_beta_with_{a}_assets_and_{w}_wage.png'.format(a=a, w=w))
            plt.close()


def main():
    steady_state_by_beta()
    reservation_wage_by_beta()
    h_by_beta()
    v_by_beta()
    savings_by_alpha()
    unsavings_by_alpha()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    main()
