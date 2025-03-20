# ---------------------------------------------------------------------------------------------
#   l4-q5, Numerical integration for S = \int_0^1 [ln(1 + x) / (1 + x^2)] dx
#       Using adaptive Simpson method and Romberg method
#   Author: Yi-Ming Ding
#   Updated: Mar 12, 2025
# ---------------------------------------------------------------------------------------------
from numpy import log
import numpy as np
from l4_q3 import trapezoidal_rule

f = lambda x: log(1 + x) / (1 + x * x)

def simpson_rule(f, a, b):
    c = (a + b) / 2
    h = (b - a) / 6
    return h * (f(a) + 4 * f(c) + f(b))

# recursive function
def adaptive_simpson(f, a, b, tol=1e-7):
    S = simpson_rule(f, a, b)  # the full integral
    c = (a + b) / 2
    S_left = simpson_rule(f, a, c)  # integrating in the left interval
    S_right = simpson_rule(f, c, b)  # integrating in the right interval

    if abs(S_left + S_right - S) < tol:
        return S_left + S_right + (S_left + S_right - S)
    else:
        return adaptive_simpson(f, a, c, tol / 2) + adaptive_simpson(f, c, b, tol / 2)  # continue the recursion

def romberg_method(f, a, b, max_iter=10, tol=1e-7):
    R = np.zeros((max_iter, max_iter))  # Romberg tabel

    for i in range(max_iter):
        n = 2 ** i  # equally division into 2^i intervals
        R[i, 0] = trapezoidal_rule(f, a, b, n)

        # consider the iteration
        for j in range(1, i + 1):
            R[i, j] = (4 ** j * R[i, j - 1] - R[i - 1, j - 1]) / (4 ** j - 1)

        # if it converges, return the results
        if i > 0 and abs(R[i, i] - R[i - 1, i - 1]) < tol:
            return R[i, i]

    return R[max_iter - 1, max_iter - 1]

if __name__ == "__main__":
    x_data = np.linspace(0, 1, 100)
    y_data = np.array([f(x) for x in x_data])

    a, b = 0, 1
    result_adaptive_simpson = adaptive_simpson(f, a, b)
    print(f"Result (adaptive Simpson) = {result_adaptive_simpson}")

    result_romberg = romberg_method(f, a, b)
    print(f"Result (romberg method) = {result_romberg}")