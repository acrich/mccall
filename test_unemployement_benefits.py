import sys
import numpy as np
from separations import binomial_draws
import matplotlib.pyplot as plt
from numba import njit
from model import Model
from wage_distribution import lognormal_draws
from agent import generate_lifetime


np.set_printoptions(threshold=sys.maxsize)


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
        reservation_wage = []
        wages_per_z.append(np.mean(np.asarray(wages)))
        assets_per_z.append(np.mean(assets))
        reservation_wages.append(np.argwhere(accept_or_reject[0, :] == 1)[0][0])
        savings_top_left.append(a_opt_employed[1, 1])
        savings_top_right.append(a_opt_employed[1, m.w_size - 1])
        savings_bottom_left.append(a_opt_employed[m.a_size - 1, 0])
        savings_bottom_right.append(a_opt_employed[m.a_size - 1, m.w_size - 1])
        savings_middle.append(a_opt_employed[50, 50])

    print(reservation_wages)

    fig, ax = plt.subplots()
    ax.set_xlabel('unemployment benefits')
    ax.set_ylabel('reservation wage')

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

    ax.plot(z_choices, wages_per_z, '-', alpha=0.4, color="C1", label=f"$mean asset level at T$")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('savings top left')
    ax.set_ylabel('assets')

    ax.plot(z_choices, savings_top_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('savings top right')
    ax.set_ylabel('assets')

    ax.plot(z_choices, savings_top_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('savings bottom left')
    ax.set_ylabel('assets')

    ax.plot(z_choices, savings_top_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('savings bottom right')
    ax.set_ylabel('assets')

    ax.plot(z_choices, savings_top_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('savings middle')
    ax.set_ylabel('assets')

    ax.plot(z_choices, savings_top_left, '-', alpha=0.4, color="C1", label=f"")
    plt.show()

if __name__ == '__main__':
    test()
