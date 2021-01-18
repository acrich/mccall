import time
from interpolation import interp
import numpy as np
from numba import njit, prange, float64, int64, objmode
from numba.experimental import jitclass
import random
from wage_distribution import lognormal_draws, μ, σ


VERY_SMALL_NUMBER = -1e+3


mccall_data = [
    ('z', float64),  # unemployment benefits
    ('β', float64),  # discount factor
    ('T', int64),  # number of periods
    ('α', float64),  # separation rate
    ('τ', float64),  # tax rate on capital (currently not in use)
    ('μ', float64),  # mean of the wage distribution
    ('σ', float64),  # standard deviation of the wage distribution
    ('ism', float64),  # inter-temporal savings motive
    ('c_hat', float64),  # minimal consumption
    ('r', float64),  # interest on assets
    ('w_min', float64),  # minimum of wage grid
    ('a_min', float64),  # minimum of assets grid
    ('a_max', float64),  # maximum of assets grid
    ('w_size', int64),  # size of wage grid
    ('w_grid', float64[:]),  # wage grid
    ('w_draws', float64[:, :]),  # 4,000 draws from the wage distribution, per existing wage
    ('a_size', int64),  # size of the assets grid
    ('a_grid', float64[:]),  # assets grid
    ('ρ', float64),  # coefficient of relative risk aversion
    ('wage_dispersion', float64),  # variance of wage offers, given previous wage
]


@jitclass(mccall_data)
class Model:
    """
    implements McCall using Monte Carlo approximation.

    With savings allowed, the equations from:
    https://python.quantecon.org/mccall_model_with_separation.html#The-Bellman-Equations

    become:

    v(w_e, a) = u(c_e) + \beta [(1 - \alpha) v(w_e, a') + \alpha \sum_{w'} h(w', a') q(w')] \; s.t. \; c_e + c_hat + a' = a + w_e
    h(w, a) = \max \{ v(w, a), u(c_u) + \beta \sum_{w'} h(w', a') q(w') \} \; s.t. \; c_u + c_hat + a' = a + z

    and equations (5) and (6) become:

    d(a') = \sum_{w'} \{ v(w', a'), u(a' - a'' + z - c_hat) +\beta d(a'') \} q(w')
    v(w,a) = u(a + w - a' - c_hat) + \beta [(1 - \alpha) v(w, a') + \alpha d(a')]
    """

    def __init__(
            self,
            z=0,
            β=0.94,
            T=408,
            α=1/34,
            τ=0.8,
            μ=μ,
            σ=σ,
            ism=1.001,
            c_hat=2,
            ρ=0.3,
            wage_dispersion=3
        ):

        # unemployment benefits
        self.z = z

        # discount factor
        self.β = β

        # see: https://www.bls.gov/news.release/pdf/nlsoy.pdf for calibration choices.
        # number of periods
        self.T = T
        # separation rate
        self.α = α

        # tax rate on capital (currently not in use)
        self.τ = τ

        # mean of the wage distribution
        self.μ = μ

        # standard deviation of the wage distribution
        self.σ = σ

        # a set of 1000 random wage draws, used to derive the expected wage
        # that's needed to compute the next period utility from being unemployed.
        w_draws = lognormal_draws(n=1000, μ=self.μ, σ=self.σ, seed=1234)

        # wage grid parameters
        self.w_min = 1e-10
        w_max = np.max(w_draws)
        self.w_size = 100
        self.w_grid = np.linspace(self.w_min, w_max, self.w_size)

        self.a_min = 1e-10
        self.a_max = 2000
        self.a_size = 200
        # see: https://stackoverflow.com/a/62740029/1408861
        self.a_grid = self.a_max*((np.linspace(self.a_min, 1, self.a_size))**2)

        self.w_draws = np.empty((self.w_size, 400))
        for j, w in enumerate(self.w_grid):
            self.w_draws[j, :] = np.abs(np.random.normal(w, self.wage_dispersion, 400))

        # minimal consumption per period
        self.c_hat = c_hat

        # interest rate on assets, given a value of the inter-temporal savings motive
        # eventually, we divide by 12 to get a monthly return on assets.
        self.ism = ism
        self.r = ((ism/self.β) - 1)

        # coefficient of relative risk aversion (only used when u() is overriden)
        self.ρ = ρ

        # wage offers are persistent, meaning they follow an AR(1) process
        # where the error term is defined by the following parameter
        self.wage_dispersion = wage_dispersion

    def u(self, c):
        if self.ρ == 1:
            return np.log(c)
        # the iso-elastic utility function is CRRA, (and thus, DARA) and satisfies prudence.
        # because of prudence and the risk of unemployment, we expect some positive level of savings.
        return (c**(1 - self.ρ) - 1) / (1 - self.ρ)

    def get_wage_offer(self, w_t):
        # persistence in wage offers through an AR(1) process
        # wage dispersion should really be the degrees of freedom in a Student's t distribution.
        # the variance defines the support, which in a mean preserving spread should be identical.
        w_t = np.random.normal(w_t, self.wage_dispersion, 1)[0]
        if w_t < 0:
            w_t = abs(w_t)
        elif w_t == 0:
            w_t = 0.1
        return w_t

    def update_d(self, h):
        d = np.empty_like(h)
        for i, a in enumerate(self.a_grid):
            hf = lambda x: interp(self.w_grid, h[i, :], x)
            for j, w in enumerate(self.w_grid):
                d[i, j] = np.mean(hf(self.w_draws[j]))
        for i in range(self.a_size):
            for j in range(self.w_size):
                if np.isnan(d[i, j]):
                    print(i)
                    print(j)
                    print(h)
                    raise Exception("d is NaN")
        return d

    def update_v(self, v, h, d):
        a_grid = self.a_grid
        v_new = np.empty((self.a_size, self.w_size))
        a_opt_employed = np.empty((self.a_size, self.w_size))
        a_opt_employed = a_opt_employed.astype(int64)
        for i, a in enumerate(a_grid):
            for j, w in enumerate(self.w_grid):
                consumption = w + a_grid[i]*(1 + self.r) - a_grid - self.c_hat

                negative_elements = np.where(consumption < 0)[0]
                if len(negative_elements) > 0:
                    max_index = negative_elements[0] # index_of_first_negative_element
                else:
                    max_index = self.a_size

                if max_index == 0:
                    # all consumption choices are negative.
                    v_new[i, j] = np.nan
                    a_opt_employed[i, j] = 0
                else:
                    rhs = self.u(consumption[:max_index]) + self.β*((1 - self.α)*v[:, j][:max_index] + self.α*d[:max_index, j])
                    v_new[i, j] = np.nanmax(rhs)
                    if np.isnan(v_new[i, j]):
                        # @TODO if nan then raise, don't assign nan above, using VERY_SMALL_NUMBER instead.
                        a_opt_employed[i, j] = 0
                    else:
                        a_opt_employed[i, j] = np.where(rhs == v_new[i, j])[0][0]

        return v_new, a_opt_employed

    def update_h(self, v, h, d, a_opt_employed):
        a_grid = self.a_grid
        h_new = np.empty((self.a_size, self.w_size))
        a_opt_unemployed = np.empty((self.a_size, self.w_size))
        a_opt_unemployed = a_opt_unemployed.astype(int64)
        accept_or_reject = np.empty((self.a_size, self.w_size))

        for i in range(self.a_size):
            consumption = self.z + a_grid[i]*(1 + self.r) - a_grid - self.c_hat

            negative_elements = np.where(consumption < 0)[0]
            if len(negative_elements) > 0:
                max_index = negative_elements[0] # index_of_first_negative_element
            else:
                max_index = self.a_size

            for j in range(self.w_size):
                if max_index == 0:
                    # all consumption choices are negative.
                    unemployment_opt = VERY_SMALL_NUMBER
                    a_opt = 0
                else:
                    unemployment = self.u(consumption[:max_index]) + self.β*d[:max_index, j]
                    unemployment_opt = np.max(unemployment)
                    a_opt = np.argmax(unemployment)

                rhs = np.asarray([unemployment_opt, v[i, j]])
                h_new[i, j] = np.nanmax(rhs)
                accept_or_reject[i, j] = np.where(rhs == h_new[i, j])[0][0]

                if accept_or_reject[i, j] == 0:
                    a_opt_unemployed[i, j] = a_opt
                else:
                    a_opt_unemployed[i, j] = a_opt_employed[i, j]
        return h_new, a_opt_unemployed, accept_or_reject


    def update(self, v, h):
        """ iterate once given existing v and h """
        #with objmode(time1='f8'):
        #    time1 = time.perf_counter()
        d = self.update_d(h)
        #with objmode():
        #    print('time: {}'.format(time.perf_counter() - time1))

        #with objmode(time1='f8'):
        #    time1 = time.perf_counter()
        v_new, a_opt_employed = self.update_v(v, h, d)
        #with objmode():
        #    print('time: {}'.format(time.perf_counter() - time1))

        #with objmode(time1='f8'):
        #    time1 = time.perf_counter()
        h_new, a_opt_unemployed, accept_or_reject = self.update_h(v, h, d, a_opt_employed)
        #with objmode():
        #    print('time: {}'.format(time.perf_counter() - time1))

        return v_new, h_new, accept_or_reject, a_opt_unemployed, a_opt_employed


    def solve_model(self, tol=1e-3, max_iter=2000):
        """
        Iterates to convergence on the Bellman equations
        """

        v = np.ones((self.a_size, self.w_size))  # Initial guess of v
        h = np.ones((self.a_size, self.w_size))  # Initial guess of h
        i = 0
        error = tol + 1

        while error > tol and i < max_iter:
            v_new, h_new, accept_or_reject, a_opt_unemployed, a_opt_employed = self.update(v, h)
            error_1 = np.nanmax(np.nanmax(np.abs(v_new - v)))
            error_2 = np.nanmax(np.nanmax(np.abs(h_new - h)))
            error = max(error_1, error_2)
            v = v_new
            h = h_new
            i += 1

            if i == max_iter:
                raise Exception("Reached max_iter without convergence")

        return v, h, accept_or_reject, a_opt_unemployed, a_opt_employed
