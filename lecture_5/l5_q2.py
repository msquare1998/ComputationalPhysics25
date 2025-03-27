# -----------------------------------------------------------------------------------------------------
#   l1-q2, Runge-Kutta method for solving y' = x \sqrt{y}, with y(2) = 4
#   Author: Yi-Ming Ding
#   Updated: Mar 19, 2025
#   Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp   # "ivp" stands for "initial value problems of ODEs"

f = lambda x, y: x * np.sqrt(y)     # the differential equation
x0, y0 = 2, 4   # Initial condition
x_range = (x0, 5)  # Solve from x0 to x=5
y_exact = lambda x: (x * x / 4 + 1) ** 2

if __name__ == "__main__":
    solution = solve_ivp(f, x_range, [y0], method='RK45', dense_output=True)   # "dense_output" for continuous solution

    x_vals = np.linspace(x0, 5, 100)
    y_vals = solution.sol(x_vals)   # generate the result

    plt.plot(x_vals, [y_exact(x) for x in x_vals], label="Exact solution")
    plt.plot(x_vals, y_vals.T, label="Runge-Kutta method", linestyle=':')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
