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


def test():
    z_choices = np.linspace(0, 10, 20)
    reservation_wages = []
    wages_per_z = []
    assets_per_z = []
    savings_top_left = []
    savings_top_right = []
    savings_bottom_left = []
    savings_bottom_right = []
    savings_middle = []
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
        savings_top_left.append(a_opt_employed[1, 1])
        savings_top_right.append(a_opt_employed[1, m.w_size - 1])
        savings_bottom_left.append(a_opt_employed[m.a_size - 1, 0])
        savings_bottom_right.append(a_opt_employed[m.a_size - 1, m.w_size - 1])
        savings_middle.append(a_opt_employed[50, 50])

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('reservation wage (with no assets)')

    ax.plot(z_choices, reservation_wages, '-', alpha=0.4, color="C1", label=f"$reservation wage with no assets$")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('wages')

    ax.plot(z_choices, wages_per_z, '-', alpha=0.4, color="C1", label=f"$mean wage$")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('assets')

    ax.plot(z_choices, assets_per_z, '-', alpha=0.4, color="C1", label=f"$mean asset level at T$")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('next period assets index for agent with no assets and low wage')

    ax.plot(z_choices, savings_top_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('next period assets index for agent with no assets and right wage')

    ax.plot(z_choices, savings_top_right, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('next period assets index for agent with high assets and low wage')

    ax.plot(z_choices, savings_bottom_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('next period assets index for agent with high assets and high wage')

    ax.plot(z_choices, savings_bottom_right, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('next period assets index for agent with medium assets and medium wage')

    ax.plot(z_choices, savings_middle, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

if __name__ == '__main__':
    test()
