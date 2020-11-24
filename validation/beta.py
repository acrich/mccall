import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from steady_state import get_steady_state
from agent import find_nearest_index


"""
higher beta means a_opt_employed and a_opt_unemployed should both be higher.
"""


DIR = '/home/shay/projects/quantecon/results/beta/'


def steady_state_by_beta():
    m = Model()
    w_choices = [4, 6, 10, 14]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    beta_choices = np.linspace(0.8, 0.99, 10)
    steady_states = np.empty((len(w_choice_indices), len(beta_choices)))
    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)
            np.save('npy/h_at_{beta}_beta.npy'.format(beta=beta), h)
            np.save('npy/v_at_{beta}_beta.npy'.format(beta=beta), v)

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            steady_states[w_choice_index, beta_index] = get_steady_state(a_opt_employed[:, w_grid_index])

    for w_choice_index, w_grid_index in enumerate(w_choice_indices):
        w = m.w_grid[w_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('discount factor β')
        ax.set_ylabel('steady-state assets')
        ax.plot(beta_choices, steady_states[w_choice_index, :], '-', alpha=0.4, color="C7", label="steady state assets")
        plt.savefig(DIR + 'steady_state_by_beta_at_{w}_wage.png'.format(w=round(w)))
        plt.close()



def savings_by_beta():
    """ i don't trust steady-states because there are too many of them. """
    m = Model()
    w_choices = [0, 34]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    beta_choices = np.array([0.8, 0.99])
    savings = np.empty((len(w_choice_indices), len(beta_choices), m.a_size))
    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            a_opt_employed = np.load('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)
            np.save('npy/h_at_{beta}_beta.npy'.format(beta=beta), h)
            np.save('npy/v_at_{beta}_beta.npy'.format(beta=beta), v)

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, beta_index, :] = a_opt_employed[:, w_grid_index]

    for beta_index, beta in enumerate(beta_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, m.a_grid, '-', alpha=0.4, color="C1", label="a")
            ax.plot(m.a_grid, savings[w_choice_index, beta_index, :], '-', alpha=0.4, color="C2", label="a'")
            ax.legend(loc="lower right")
            plt.savefig(DIR + 'savings_at_{w}_wage_and_{beta}_beta.png'.format(w=round(w), beta=str(beta).replace('.', '_')))
            plt.close()


def unsaving_by_beta():
    m = Model()
    w_choice_indices = np.array([0, 50])
    beta_choices = np.array([0.8, 0.99])
    savings = np.empty((len(w_choice_indices), len(beta_choices), m.a_size))
    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            a_opt_unemployed = np.load('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)
            np.save('npy/h_at_{beta}_beta.npy'.format(beta=beta), h)
            np.save('npy/v_at_{beta}_beta.npy'.format(beta=beta), v)

        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            savings[w_choice_index, beta_index, :] = a_opt_unemployed[:, w_grid_index]

    for beta_index, beta in enumerate(beta_choices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('current period assets')
            ax.set_ylabel('next period assets')
            ax.plot(m.a_grid, m.a_grid, '-', alpha=0.4, color="C1", label="a")
            ax.plot(m.a_grid, savings[w_choice_index, beta_index], '-', alpha=0.4, color="C1", label="a'")
            ax.legend(loc="lower right")
            plt.savefig(DIR + 'unsavings_at_{w}_wage_and_{beta}_beta.png'.format(w=round(w), beta=str(beta).replace('.', '_')))
            plt.close()


def reservation_wage_by_beta():
    m = Model()
    beta_choices = np.linspace(0.5, 0.99, 10)
    a_choices = [10]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    reservation_wages = np.empty((len(a_choice_indices), len(beta_choices)))

    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            accept_or_reject = np.load('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)
            np.save('npy/h_at_{beta}_beta.npy'.format(beta=beta), h)
            np.save('npy/v_at_{beta}_beta.npy'.format(beta=beta), v)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, beta_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('discount factor β')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=round(a)))
        ax.plot(beta_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C3", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_beta_with_{a}_assets.png'.format(a=round(a)))
        plt.close()


def h_by_beta():
    m = Model()
    beta_choices = np.linspace(0.8, 0.99, 10)
    w_choices = [0, 10]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    a_choices = [0, 51]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    h_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(beta_choices)))

    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            h = np.load('npy/h_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)
            np.save('npy/h_at_{beta}_beta.npy'.format(beta=beta), h)
            np.save('npy/v_at_{beta}_beta.npy'.format(beta=beta), v)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                h_results[a_choice_index, w_choice_index, beta_index] = h[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('discount factor β')
            ax.set_ylabel('h (utility in unemployment) with {a} assets and {w} wage'.format(a=round(a), w=round(w)))
            ax.plot(beta_choices, h_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C5", label=f"")
            plt.savefig(DIR + 'h_per_beta_with_{a}_assets_and_{w}_wage.png'.format(a=round(a), w=round(w)))
            plt.close()


def v_by_beta():
    m = Model()
    beta_choices = np.linspace(0.8, 0.99, 10)
    w_choices = [0, 10]
    w_choice_indices = np.asarray([find_nearest_index(m.w_grid, w) for w in w_choices])
    a_choices = [0, 51]
    a_choice_indices = np.asarray([find_nearest_index(m.a_grid, a) for a in a_choices])
    v_results = np.empty((len(a_choice_indices), len(w_choice_indices), len(beta_choices)))

    for beta_index, beta in enumerate(beta_choices):
        m = Model(β=beta)
        try:
            v = np.load('npy/v_at_{beta}_beta.npy'.format(beta=beta))
        except IOError:
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            np.save('npy/a_opt_employed_at_{beta}_beta.npy'.format(beta=beta), a_opt_employed)
            np.save('npy/a_opt_unemployed_at_{beta}_beta.npy'.format(beta=beta), a_opt_unemployed)
            np.save('npy/accept_or_reject_at_{beta}_beta.npy'.format(beta=beta), accept_or_reject)
            np.save('npy/h_at_{beta}_beta.npy'.format(beta=beta), h)
            np.save('npy/v_at_{beta}_beta.npy'.format(beta=beta), v)

        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                v_results[a_choice_index, w_choice_index, beta_index] = v[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('discount factor β')
            ax.set_ylabel('v (utility in employment) with {a} assets and {w} wage'.format(a=round(a), w=round(w)))
            ax.plot(beta_choices, v_results[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C6", label=f"")
            plt.savefig(DIR + 'v_per_beta_with_{a}_assets_and_{w}_wage.png'.format(a=round(a), w=round(w)))
            plt.close()


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    steady_state_by_beta()
    reservation_wage_by_beta()
    h_by_beta()
    v_by_beta()
    savings_by_beta()
    unsaving_by_beta()


if __name__ == '__main__':
    main()
