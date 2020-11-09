import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

N_points = 4000

μ=0.8
σ=0.7

def lognormal_draws(n=100, μ=1.5, σ=1.4, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z) * 2
    return w_draws


def bimodal_draws(n=100, μ=1.5, σ=1.4, seed=1234):
    mu, sigma = 1, 0.7 # mean and standard deviation
    low_w_draws = np.random.normal(mu, sigma, n)
    low_w_draws = lognormal_draws(n=n, μ=0.5, σ=0.2, seed=seed)
    mu, sigma = 50, 1.7 # mean and standard deviation
    high_w_draws = np.random.normal(mu, sigma, n)
    α = 0.05
    draws = bernoulli.rvs(1 - α, size=n)
    w_draws = (1-draws)*high_w_draws + draws*low_w_draws
    return w_draws


if __name__ == '__main__':
    w = bimodal_draws(n=N_points, μ=μ, σ=σ)
    average = np.mean(w)
    print(np.min(w))
    print(average)
    print(np.median(w))
    print(np.max(w))

    count, bins, ignored = plt.hist(w, 100, density=True)
    plt.show()
