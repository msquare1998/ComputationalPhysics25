# ------------------------------------------------------------------------------------------------------------
#   l3-q2, Lagrange interpolation
#   Author: Yi-Ming Ding
#   Updated: Mar 5, 2025
#   Reference： https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

x_nodes = np.random.uniform(0, 2 * np.pi, 9)     # randomly generate 9 points
y_nodes = np.sin(x_nodes)

if __name__ == "__main__":
    print(f"x_nodes = {x_nodes}")
    print(f"y_nodes = {y_nodes}")
    poly = lagrange(x_nodes, y_nodes)
    print("Lagrange Polynomial:", poly)

    x_vals = np.linspace(0, 2 * np.pi, 21)
    y_vals = poly(x_vals)

    plt.title("Lagrange Interpolation of $y = sin(x)$")
    plt.plot(x_vals, y_vals, label="Lagrange Interpolation")
    plt.scatter(x_nodes, y_nodes, color='red', label="Data Points")
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.show()