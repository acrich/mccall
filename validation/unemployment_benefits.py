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
a higher c (unemployment benefits), means:
in times of unemployment, consumption grows, both because of the benefits themselves and because of lower savings (not sure about this because savings doesn't grow in unemployment).
in times of employment, savings should be lower, meaning consumption again grows.
reservation wage should rise with c. should it do so for every savings level? probably the effect should decrease in savings. as a result, average wage should increase, and wage disparity should decrease.

we see reservation wage and average wages growing with benefits.
assets are unclear, it depends on the current wage and existing assets.
benefits above a certain threshold nullify the savings motive, because ism<1. we see this in the savings graphs.
"""


DIR = '/home/shay/projects/quantecon/results/unemployment_benefits/'


# a_opt_unemployed by assets, given several levels of benefits
# a_opt_employed by assets, given several levels of benefits (should stay the same)
def test_savings():
    z_choices = np.linspace(0, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 10, 2)
    savings_employed = np.empty((len(a_choice_indices), len(w_choice_indices), len(z_choices)))
    savings_unemployed = np.empty((len(a_choice_indices), len(w_choice_indices), len(z_choices)))
    for z_choice_index, z in enumerate(z_choices):
        m = Model(z=z)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings_employed[a_choice_index, w_choice_index, z_choice_index] = a_opt_employed[a_grid_index, w_grid_index]
                savings_unemployed[a_choice_index, w_choice_index, z_choice_index] = a_opt_unemployed[a_grid_index, w_grid_index]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            fig, ax = plt.subplots()
            ax.set_xlabel('unemployment benefits')
            ax.set_ylabel('next period assets index for agent with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(z_choices, savings_employed[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C1", label=f"savings when employed")
            ax.plot(z_choices, savings_unemployed[a_choice_index, w_choice_index, :], '-', alpha=0.4, color="C2", label=f"savings when unemployed")
            plt.savefig(DIR + 'savings_per_c_hat_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def test_benefits():
    z_choices = np.linspace(0, 10, 20)
    reservation_wages = []
    wages_per_z = []
    assets_per_z = []
    for z in z_choices:
        m = Model(z=z)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        wages = []
        assets = []
        T = 100
        for i in range(1000):
            a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(T=T, a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)
            wages.append(np.dot(realized_wage,employment_spells)/np.sum(employment_spells))
            assets.append(a[T-1])
        wages_per_z.append(np.mean(np.asarray(wages)))
        assets_per_z.append(np.mean(assets))
        reservation_wages.append(np.argwhere(accept_or_reject[0, :] == 1)[0][0])

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('reservation wage (with no assets)')

    ax.plot(z_choices, reservation_wages, '-', alpha=0.4, color="C1", label=f"$reservation wage with no assets$")
    plt.savefig(DIR + 'reservation_wage_by_benefits_no_assets.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('wages')

    ax.plot(z_choices, wages_per_z, '-', alpha=0.4, color="C1", label=f"$mean wage$")
    plt.savefig(DIR + 'wage_by_benefits.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('assets')

    ax.plot(z_choices, assets_per_z, '-', alpha=0.4, color="C1", label=f"$mean asset level at T$")
    plt.savefig(DIR + 'assets_at_T_by_benefits.png')
    plt.close()


def test():
    test_savings()
    test_benefits()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    test()
