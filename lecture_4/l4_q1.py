# ---------------------------------------------------------------------------------------------
#   l4-q1, Numerical differentiation
#   Author: Yi-Ming Ding
#   Updated: Mar 12, 2025
# ---------------------------------------------------------------------------------------------
from jax.numpy import sin, pi
from jax import grad
import matplotlib.pyplot as plt
import numpy as np

n = 40
f = lambda x: sin(x)
dx = 1e-4
f_prime = lambda x: (f(x + dx) - f(x - dx)) / (2 * dx)
f_prime_ad = lambda x: grad(f)(x)
a, b = 0, 2 * pi

if __name__ == "__main__":
    x_data = np.linspace(0, 2 * np.pi, n)   # generate x_data
    plt.plot(x_data, f_prime(x_data), marker="o", linestyle="", label="Numerical differentiation")
    plt.plot(x_data, [f_prime_ad(x) for x in x_data], marker="x", linestyle="", label="Automatic differentiation")
    plt.legend()
    plt.show()
