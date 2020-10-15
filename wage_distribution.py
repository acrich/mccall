import numpy as np
import matplotlib.pyplot as plt


N_points = 100000


def lognormal_draws(n=100, μ=1.5, σ=1.4, seed=1234):
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws


w = lognormal_draws(n=N_points, μ=0.4, σ=0.6)
average = np.mean(w)
print(average)
bins = np.linspace(0, 10, 1000)

fig,ax = plt.subplots(1,1)
ax.hist(w, bins = bins)
plt.axvline(x=average)
ax.set_title("histogram of wage distribution")
ax.set_xticks(bins)
ax.set_xlabel('wage')
ax.set_ylabel('density')
plt.show()
