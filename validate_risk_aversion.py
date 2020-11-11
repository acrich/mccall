import os
from model_with_risk import Model
import numpy as np
import matplotlib.pyplot as plt


"""
if we use a ces utility where we can control the relative/absolute risk aversion.
the higher the ARA, the lower the utility will be when we increase the either the wage variation or alpha.
higher ARA means we have less motivation to increase consumption, so we should save more.

measuring it the other way around, higher consumption means lower A(c). The higher the ARA, the higher the effect will be.
there should be an indirect effect of wages/savings/benefits on consumption and through that on A(c).

higher ARA means lower reservation wage.

our function should withhold DARA and consequently also prudence.
however, relative risk aversion need not be constant. all options are legitimate.
there's evidence that relative risk aversion is U shaped.
"""

DIR = '/home/shay/projects/quantecon/results/risk_aversion/'


def test_consumption():
    # consumption should decrease when risk aversion increases
    rho_choices = np.linspace(0.5, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 10, 2)

    consumption = np.empty((len(a_choice_indices), len(w_choice_indices), len(rho_choices)))
    for rho_index, rho in enumerate(rho_choices):
        m = Model(ρ=rho)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                a_prime = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]
                a = m.a_grid[a_grid_index]
                w = m.w_grid[w_grid_index]
                c = (1 + m.r)*a + w - m.c_hat - a_prime
                consumption[a_choice_index, w_choice_index, rho_index] = c

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('coefficient of relative risk aversion ρ')
            ax.set_ylabel('consumption with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(rho_choices, consumption[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'consumption_per_rho_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def test_savings():
    # savings should increase when risk aversion increases
    rho_choices = np.linspace(0.5, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 10, 2)

    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(rho_choices)))
    for rho_index, rho in enumerate(rho_choices):
        m = Model(ρ=rho)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, rho_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('coefficient of relative risk aversion ρ')
            ax.set_ylabel('savings with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(rho_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'savings_per_rho_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def test_reservation_wage():
    # as risk aversion increases, reservation wage decreases
    rho_choices = np.linspace(0.5, 10, 20)
    a_choice_indices = np.arange(0, 30, 3)

    reservation_wages = np.empty((len(a_choice_indices), len(rho_choices)))
    for rho_index, rho in enumerate(rho_choices):
        m = Model(ρ=rho)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, rho_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('coefficient of relative risk aversion ρ')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(rho_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_rho_with_{a}_assets.png'.format(a=a))
        plt.close()

def test_reservation_wage_and_ism():
    # ism should interact with ARA. if ism is higher (interest is higher), we'll take more risks in order to increase savings.
    # so the reservation wage should increase with ism, and higher ARA should make the effect smaller.
    ism_choices = np.linspace(0.5, 1.5, 10)
    rho_choices = np.linspace(0.5, 10, 20)
    a_choice_indices = np.arange(0, 30, 3)

    reservation_wages = np.empty((len(a_choice_indices), len(rho_choices), len(ism_choices)))
    for ism_index, ism in enumerate(ism_choices):
        for rho_index, rho in enumerate(rho_choices):
            m = Model(ism=ism, ρ=rho)
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            for a_choice_index, a_grid_index in enumerate(a_choice_indices):
                reservation_wages[a_choice_index, rho_index, ism_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for rho_index, rho in enumerate(rho_choices):
            a = m.a_grid[a_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('inter-temporal savings motive')
            ax.set_ylabel('reservation wage with {rho} rho and {a} assets'.format(rho=rho, a=a))
            ax.plot(ism_choices, reservation_wages[a_choice_index, rho_index, :], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'reservation_wage_per_ism_with_{rho}_rho_and_{a}_assets.png'.format(rho=rho, a=a))
            plt.close()


def test_wage_variation():
    # an increase in risk aversion would make utility drop harder when wage variation increases
    # it makes more sense to look at savings decisions than cardinal utilities, so we'll do that.
    rho_choices = np.linspace(0.5, 10, 6)
    mu_choices = np.linspace(0.1, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 6, 2)

    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(rho_choices), len(mu_choices)))
    for rho_index, rho in enumerate(rho_choices):
        for mu_index, mu in enumerate(mu_choices):
            m = Model(ρ=rho, μ=mu)
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            for a_choice_index, a_grid_index in enumerate(a_choice_indices):
                for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                    savings[a_choice_index, w_choice_index, rho_index, mu_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            for rho_index, rho in enumerate(rho_choices):
                a = m.a_grid[a_grid_index]
                w = m.w_grid[w_grid_index]
                fig, ax = plt.subplots()
                ax.set_xlabel('wage variation μ')
                ax.set_ylabel('next period assets for agent with {rho} risk aversion {a} assets and {w} wage'.format(rho=rho, a=a, w=w))
                ax.plot(mu_choices, savings[a_choice_index, w_choice_index, rho_index, :], '-', alpha=0.4, color="C1", label=f"")
                plt.savefig(DIR + 'savings_per_mu_with_{rho}_rho_and_{w}_wage_and_{a}_assets.png'.format(rho=rho, a=a, w=w))
                plt.close()


def test_separations_rate():
    # an increase in risk aversion would make utility drop harder when alpha increases
    # again, looking at savings decisions instead of utilities.
    alpha_choices = np.linspace(0.05, 0.5, 6)
    mu_choices = np.linspace(0.1, 10, 20)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 6, 2)

    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(alpha_choices), len(mu_choices)))
    for alpha_index, alpha in enumerate(alpha_choices):
        for mu_index, mu in enumerate(mu_choices):
            m = Model(α=alpha, μ=mu)
            v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
            for a_choice_index, a_grid_index in enumerate(a_choice_indices):
                for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                    savings[a_choice_index, w_choice_index, alpha_index, mu_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            for alpha_index, alpha in enumerate(alpha_choices):
                a = m.a_grid[a_grid_index]
                w = m.w_grid[w_grid_index]
                fig, ax = plt.subplots()
                ax.set_xlabel('separation rate α')
                ax.set_ylabel('next period assets for agent with {alpha} risk aversion {a} assets and {w} wage'.format(alpha=alpha, a=a, w=w))
                ax.plot(mu_choices, savings[a_choice_index, w_choice_index, alpha_index, :], '-', alpha=0.4, color="C1", label=f"")
                plt.savefig(DIR + 'savings_per_mu_with_{alpha}_alpha_and_{w}_wage_and_{a}_assets.png'.format(alpha=alpha, a=a, w=w))
                plt.close()


def test():
    test_consumption()
    test_savings()
    test_reservation_wage()
    test_reservation_wage_and_ism()
    test_wage_variation()
    test_separations_rate()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    test()
