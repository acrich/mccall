import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/shay/projects/quantecon')
from model import Model
from agent import generate_lifetime


"""
a, u_t, consumption and realized_wage are results of decisions in a path over time.
they act differently than the direct results of solving the model.
essentially, we should see the following:
a increases when employment_spells is set, and decreases when it's not.
a should move together with realized_wage
a' and consumption should match a and realized_wage, at every period.
at higher a, consumption smoothing should happen more.
what we're really smoothing is not consumption but intertemporal utility, the value of v/h should be smooth over time, while u(c) may be different.
consumption should move with the realized_wage.
"""


DIR = '/home/shay/projects/quantecon/results/lifetime/'


def main():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    m = Model()
    try:
        v = np.load('npy/v.npy')
        h = np.load('npy/h.npy')
        accept_or_reject = np.load('npy/accept_or_reject.npy')
        a_opt_unemployed = np.load('npy/a_opt_unemployed.npy')
        a_opt_employed = np.load('npy/a_opt_employed.npy')
    except IOError:
        v, h, accept_or_reject, a_opt_unemployed, a_opt_employed = m.solve_model()
        np.save('npy/v.npy', v)
        np.save('npy/h.npy', h)
        np.save('npy/accept_or_reject.npy', accept_or_reject)
        np.save('npy/a_opt_unemployed.npy', a_opt_unemployed)
        np.save('npy/a_opt_employed.npy', a_opt_employed)

    a, u_t, realized_wage, employment_spells, consumption, separations, reservation_wage = generate_lifetime(a_0=1, model=m, accept_or_reject=accept_or_reject, a_opt_unemployed=a_opt_unemployed, a_opt_employed=a_opt_employed)

    # a increases when employment_spells is set, and decreases when it's not.
    change_in_assets = np.empty((m.T))
    change_in_assets[0] = 0
    for t in range(1, m.T):
        change_in_assets[t] = a[t] - a[t - 1]

    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.plot(range(m.T), change_in_assets, '-', alpha=0.4, color="C3", label="change from last period assets")
    ax.plot(range(m.T), employment_spells, '-', alpha=0.4, color="C4", label="is employed")
    ax.legend(loc='lower right')
    plt.savefig(DIR + 'change_in_assets_and_employment_status.png')
    plt.close()

    # a should move together with wage
    w = realized_wage
    for t in range(m.T):
        if employment_spells[t] == 0:
            w[t] = m.z

    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.plot(range(m.T), a, '-', alpha=0.4, color="C3", label="asset level")
    ax.plot(range(m.T), w, '-', alpha=0.4, color="C4", label="wage")
    ax.legend(loc='lower right')
    plt.savefig(DIR + 'assets_and_realized_wage.png')
    plt.close()

    # a' and consumption should match a and realized_wage, at every period.
    for t in range(m.T - 1):
        market_clearance = a[t + 1] + consumption[t] - a[t]*(1 + m.r) - w[t] - m.c_hat
        try:
            assert(abs(market_clearance) < 0.1)
        except:
            print("market clearance assertion failed at period {t}.".format(t=t))
            print(market_clearance)
            print(a[t+1])
            print(consumption[t])
            print(a[t])
            print(w[t])

    # consumption should move with the realized_wage.
    fig, ax = plt.subplots()
    ax.set_xlabel('periods')
    ax.plot(range(m.T), consumption, '-', alpha=0.4, color="C3", label="consumption")
    ax.plot(range(m.T), w, '-', alpha=0.4, color="C4", label="wage")
    ax.legend(loc='lower right')
    plt.savefig(DIR + 'consumption_and_wage.png')
    plt.close()


if __name__ == '__main__':
    main()
