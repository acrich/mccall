import numpy as np
import matplotlib.pyplot as plt
from numba import njit


NUM_DRAWS = 4000
NUM_BINS = 200


# we assume a lognormal distribution of wages with median=50K and mean=80K.
# this translates to monthly mean and median of 4+1/6 and 6+2/3.
# I calculate the underlying normal distribution parameters like so:
# median = exp(μ) => ln(median) = μ
# mean = exp(μ + σ^2/2) => ln(mean) = μ + σ^2/2 => σ^2 = 2*(ln(mean) - μ)
# μ = ln(4+1/6) = 1.4271
# σ = sqrt(2*(ln(6+2/3) - ln(4+1/6))) = 0.9695

μ = 1.4271
σ = 0.9695


@njit
def lognormal_draws(n=100, μ=μ, σ=σ, seed=1234):
    np.random.seed(seed)
    s = np.random.normal(μ, σ, n)
    return np.exp(s)


def get_stats():
    w = lognormal_draws(n=NUM_DRAWS, μ=μ, σ=σ)
    return {
        "minimum": np.min(w),
        "maximum": np.max(w),
        "average": np.mean(w),
        "median": np.median(w),
    }


def gen_plot():
    w = lognormal_draws(n=NUM_DRAWS, μ=μ, σ=σ)
    count, bins, ignored = plt.hist(w, NUM_BINS, density=True)
    plt.savefig('results/wage_distribution.png')
    plt.close()


if __name__ == '__main__':
    w = lognormal_draws(n=NUM_DRAWS, μ=μ, σ=σ)
    print("minimum is {}.".format(np.min(w)))
    print("maximum is {}.".format(np.max(w)))
    print("average is {}.".format(np.mean(w)))
    print("median is {}.".format(np.median(w)))

    count, bins, ignored = plt.hist(w, NUM_BINS, density=True)
    plt.savefig('results/wage_distribution.png')
    plt.show()

    average = np.mean(w)
    below_mean = 0
    above_mean = 0
    for draw in w:
        if draw <= average:
            below_mean += 1
        else:
            above_mean += 1
    print("below mean is {}.".format(below_mean))
    print("above mean is {}.".format(above_mean))
    print("so about 70% are below the mean, which is expected")
