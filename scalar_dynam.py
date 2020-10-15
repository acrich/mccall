import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Set the axes through the origin
for spine in ['left', 'bottom']:
    ax.spines[spine].set_position('zero')
    ax.spines[spine].set_color('green')
for spine in ['right', 'top']:
    ax.spines[spine].set_color('none')

xmin, xmax = 0, 4
xgrid = np.linspace(xmin, xmax, 200)

ax.plot(xgrid,
           xgrid,
           'b-',
           alpha=0.6,
           lw=1,
           label='y=x')

A, s, alpha, delta = 2, 0.3, 0.3, 0.4
def g(k):
    return A * s * k**alpha + (1 - delta) * k

def linear(x, a, b):
    return a*x + b;

ax.plot(xgrid,
          linear(xgrid, 0.5, 1),
          'k-',
          alpha=0.6,
          lw=1,
          label='y=x')

ax.plot(xgrid,
          linear(xgrid, -0.5, 1),
          'r-',
          alpha=0.6,
          lw=1,
          label='y=x')

plt.show()

T =  50
x_t_over_time = np.zeros(T)
x_t = 0.5
a = -0.5
b = 1
for t in range(1, T):
    x_t = linear(x_t, a, b)
    x_t_over_time[t] = x_t

fig, ax = plt.subplots()
ax.plot(range(0, T),
          x_t_over_time,
          'k-',
          alpha=0.6,
          lw=1,
          label='y=x')
plt.show()
