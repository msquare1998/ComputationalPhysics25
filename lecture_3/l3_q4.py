# ------------------------------------------------------------------------------------------------------------
#   l3-q4, Cubic Spline Interpolation
#   Author: Yi-Ming Ding
#   Updated: Mar 5, 2025
#   Reference： https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

runge_function = lambda x : 1 / (1 + x * x)

n = 10
x_nodes = np.linspace(-1, 1, n)  # equally spaced points
y_nodes = runge_function(x_nodes)

if __name__ == "__main__":
    plt.plot(x_nodes, y_nodes, label="Lagrange Interpolation")
    cs = CubicSpline(x_nodes, y_nodes)

    x_vals = np.linspace(-1, 1, 100)
    y_vals = cs(x_vals)

    plt.title("Cubic Spline Interpolation of $y = 1/(1+x^2)$")
    plt.plot(x_vals, runge_function(x_vals)  , label="Original function", linestyle="-")
    plt.plot(x_vals, y_vals, label="Cubic Spline Interpolation", linestyle=':')
    plt.scatter(x_nodes, y_nodes, color='red', label="Interpolation Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()