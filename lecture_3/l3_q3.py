# ------------------------------------------------------------------------------------------------------------
#   l3-q3, Lagrange interpolation and the Runge function
#   Author: Yi-Ming Ding
#   Updated: Mar 5, 2025
#   Reference： https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

runge_function = lambda x : 1 / (1 + x * x)
n = 20
x_nodes = np.linspace(-5, 5, n) # equally spaced points
y_nodes = runge_function(x_nodes)

"""
    Runge phenomenon:
        equally spaced interpolation points make the interpolation unstable
"""

if __name__ == "__main__":
    print(f"x_nodes = {x_nodes}")
    print(f"y_nodes = {y_nodes}")
    poly = lagrange(x_nodes, y_nodes)
    print("Lagrange Polynomial:", poly)

    x_vals = np.linspace(-5, 5, 100)
    y_vals = poly(x_vals)

    plt.title("Lagrange Interpolation of $y = 1/(1+x^2)$")
    plt.plot(x_vals, runge_function(x_vals), label="Original function", linestyle="-")
    plt.plot(x_vals, y_vals, label="Lagrange Interpolation", linestyle=":")
    plt.scatter(x_nodes, y_nodes, color='red', label="Data Points")
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.show()

