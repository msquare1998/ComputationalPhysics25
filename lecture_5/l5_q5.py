# -----------------------------------------------------------------------------------------------------
#   l1-q5, Boundary value problem for
#                   -y'' + 2y / x^2  = 1 / x
#                       y(2) = 0, y(3) = 0
#   Author: Yi-Ming Ding
#   Updated: Mar 19, 2025
#   Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp   # "bvp" stands for "boundary value problem"

# the system of differential equations, similar to the last problem
def f(x, y):
    y1, y2 = y
    dy_dx = [y2, (2 * y1) / x**2 - 1 / x]
    return dy_dx

# the boundary condition
def bc(ya, yb):
    return np.array([ya[0], yb[0]])  # y(2) = 0, y(3) = 0

if __name__ == "__main__":
    x = np.linspace(2, 3, 100)
    y_init = np.zeros((2, x.size))  # some initial guess
    sol = solve_bvp(f, bc, x, y_init)

    plt.plot(sol.x, sol.y[0], label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.show()
