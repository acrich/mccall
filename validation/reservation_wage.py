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


DIR = '/home/shay/projects/quantecon/results/reservation_wage/'


"""
reservation wage per asset level - rich and poor make different decisions

accept_or_reject determines the reservation wage. for every savings level it gives a threshold of wage above which we move from reject to accept.
that's all the information that's conveyed there. we expect the reservation wage to increase with savings.
the reservation wage will increase in benefits.
it'll increase in beta.
it'll increase with the mean wage. it'll increase with the variation of the wage distribution because more risk is better in this model.
"""


def gen_plots(result_var_name, choice_var_name, num_choices, min_choice, max_choice, indices, grid, get_result, results_path):
    choices = np.linspace(min_choice, max_choice, num_choices)
    m = Model()
    a_index_choices = np.arange(0, len(grid), 10)
    results = np.empty((len(indices), num_choices))
    for choice_index, choice in enumerate(choices):
        m = Model(**{choice_var_name: choice})
        model_output = m.solve_model()
        for index_in_indices, grid_index in enumerate(indices):
            results[index_in_indices, choice_index] = get_result(grid_index, model_output)

    for index_in_indices, grid_index in enumerate(indices):
        a = grid[grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel(choice_var_name)
        ax.set_ylabel('{result_var_name} (with {a} assets)'.format(result_var_name=result_var_name, a=a))
        ax.plot(choices, results[index_in_indices, :], '-', alpha=0.4, color="C2", label=f"$$")
        plt.savefig(results_path + '{result_var_name}_by_{choice_var_name}_with_{a}_assets.png'.format(result_var_name=result_var_name, choice_var_name=choice_var_name, a=a))
        plt.close()


def assets():
    m = Model()
    v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
    reservation_wage = np.empty_like(m.a_grid)
    for i, a in enumerate(m.a_grid):
        reservation_wage[i] = np.argwhere(accept_or_reject[i, :]  == 1)[0][0]

    fig, ax = plt.subplots()
    ax.set_xlabel('assets level')
    ax.set_ylabel('reservation wage')
    ax.plot(m.a_grid, reservation_wage, '-', alpha=0.4, color="C5", label=f"$$")
    plt.savefig(DIR + 'reservation_wage_by_assets.png')
    plt.close()


def beta():
    m = Model()

    def get_result(grid_index, model_output):
        return np.argwhere(model_output[2][grid_index, :] == 1)[0][0]

    gen_plots(
        result_var_name='reservation_wage',
        choice_var_name='β',
        num_choices=20,
        min_choice=0.1,
        max_choice=0.95,
        indices=np.arange(0, m.a_size, 10),
        grid=m.a_grid,
        get_result=get_result,
        results_path=DIR
    )


def mu():
    m = Model()

    def get_result(grid_index, model_output):
        return np.argwhere(model_output[2][grid_index, :] == 1)[0][0]

    gen_plots(
        result_var_name='reservation_wage',
        choice_var_name='μ',
        num_choices=20,
        min_choice=0.1,
        max_choice=10,
        indices=np.arange(0, m.a_size, 10),
        grid=m.a_grid,
        get_result=get_result,
        results_path=DIR
    )


def sigma():
    m = Model()

    def get_result(grid_index, model_output):
        return np.argwhere(model_output[2][grid_index, :] == 1)[0][0]

    gen_plots(
        result_var_name='reservation_wage',
        choice_var_name='σ',
        num_choices=20,
        min_choice=0.1,
        max_choice=5,
        indices=np.arange(0, m.a_size, 10),
        grid=m.a_grid,
        get_result=get_result,
        results_path=DIR
    )


def benefits():
    m = Model()

    def get_result(grid_index, model_output):
        return np.argwhere(model_output[2][grid_index, :] == 1)[0][0]

    gen_plots(
        result_var_name='reservation_wage',
        choice_var_name='z',
        num_choices=20,
        min_choice=0,
        max_choice=10,
        indices=np.arange(0, 30, 3),
        grid=m.a_grid,
        get_result=get_result,
        results_path=DIR
    )


def main():
    assets()
    benefits()
    beta()
    mu()
    sigma()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    main()
