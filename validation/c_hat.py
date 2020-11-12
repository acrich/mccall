import os
from model_with_risk import Model
import numpy as np
import matplotlib.pyplot as plt


"""
increasing minimal consumption should increase consumption (diminishing effect, maybe we should see a discontinuity at the minimal consumption threshold).
you'd think it'll diminish savings, but really only up to the minimal consumption level. after that, savings should increase because of the added risk.
increase in minimal consumption should raise the reservation wage.
in times of unemployment, increased minimal consumption means faster burn rate (churn), so it diminishes the gap in behavior between rich and poor.

minimal consumption should never be below the minimum wage... and minimum wage should probably not be zero, like it is in our model so far.
that way, poor unemployed can always accept the offer to be able to pay for the minimal consumption, so you get a higher probability that they'll end up with absolutely low wages.
however, as stated above, rich households will grow poor faster, so they'll end up with lower wages as well.
we can't anticipate which effect is stronger.
however, we can anticipate higher savings (but only for the part of wages after minimal consumption has been spent).
this higher saving rate is for both types of households, but we'll see it happen more for the rich because they'll have the higher wages.

higher minimal consumption should decrease a_opt_employed when wage is low, and have no effect when wage is high.
it should increase churn when unemployed, so a_opt_unemployed may be lower for every asset level, but only when the wage is below the reservation wage.

higher minimal consumption should improve consumption smoothing for the first few unemployment periods, but worsen it for longer spells.
"""


DIR = '/home/shay/projects/quantecon/results/minimal_consumption/'


def savings():
    c_hat_choices = np.arange(0, 6)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 10, 2)

    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(c_hat_choices)))
    for c_hat_index, c_hat in enumerate(c_hat_choices):
        m = Model(c_hat=c_hat)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, c_hat_index] = m.a_grid[a_opt_employed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('minimal consumption $(\overline{c})$')
            ax.set_ylabel('savings with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(c_hat_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'savings_per_c_hat_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def reservation_wage():
    c_hat_choices = np.linspace(0.5, 10, 20)
    a_choice_indices = np.arange(0, 30, 3)

    reservation_wages = np.empty((len(a_choice_indices), len(c_hat_choices)))
    for c_hat_index, c_hat in enumerate(c_hat_choices):
        m = Model(c_hat=c_hat)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            reservation_wages[a_choice_index, c_hat_index] = np.argwhere(accept_or_reject[a_grid_index, :]  == 1)[0][0]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        a = m.a_grid[a_grid_index]
        fig, ax = plt.subplots()
        ax.set_xlabel('minimal consumption $(\overline{c})$')
        ax.set_ylabel('reservation wage with {a} assets'.format(a=a))
        ax.plot(c_hat_choices, reservation_wages[a_choice_index, :], '-', alpha=0.4, color="C1", label=f"")
        plt.savefig(DIR + 'reservation_wage_per_c_hat_with_{a}_assets.png'.format(a=a))
        plt.close()


def burn_rate():
    c_hat_choices = np.arange(0, 6)
    a_choice_indices = np.arange(0, 15, 5)
    w_choice_indices = np.arange(0, 10, 2)

    savings = np.empty((len(a_choice_indices), len(w_choice_indices), len(c_hat_choices)))
    for c_hat_index, c_hat in enumerate(c_hat_choices):
        m = Model(c_hat=c_hat)
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        for a_choice_index, a_grid_index in enumerate(a_choice_indices):
            for w_choice_index, w_grid_index in enumerate(w_choice_indices):
                savings[a_choice_index, w_choice_index, c_hat_index] = m.a_grid[a_opt_unemployed[a_grid_index, w_grid_index]]

    for a_choice_index, a_grid_index in enumerate(a_choice_indices):
        for w_choice_index, w_grid_index in enumerate(w_choice_indices):
            a = m.a_grid[a_grid_index]
            w = m.w_grid[w_grid_index]
            fig, ax = plt.subplots()
            ax.set_xlabel('minimal consumption $(\overline{c})$')
            ax.set_ylabel('burn rate with {a} assets and {w} wage'.format(a=a, w=w))
            ax.plot(c_hat_choices, savings[a_choice_index, w_choice_index], '-', alpha=0.4, color="C1", label=f"")
            plt.savefig(DIR + 'burn_rate_per_c_hat_with_{w}_wage_and_{a}_assets.png'.format(a=a, w=w))
            plt.close()


def main():
    savings()
    reservation_wage()
    burn_rate()


if __name__ == '__main__':
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    main()
