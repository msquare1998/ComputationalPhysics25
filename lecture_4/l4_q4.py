# ---------------------------------------------------------------------------------------------
#   l4-q4, Numerical integration for I = \int_0^1 [4 / (1 + x^2)] dx
#       Using composite trapezoidal rule and composite Simpson rule
#   Author: Yi-Ming Ding
#   Updated: Mar 12, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
f = lambda x: 4 / (1 + x * x)

def composite_trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))

def composite_simpson_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return (h / 3) * (f(x[0]) + 4 * np.sum(f(x[1:-1:2])) + 2 * np.sum(f(x[2:-2:2])) + f(x[-1]))

if __name__ == "__main__":
    n = 16     # number of sub-intervals
    a, b = 0, 1

    result_composite_trapezoidal = composite_trapezoidal_rule(f, a, b, n)
    print(f"result (composite trapezoidal) = {result_composite_trapezoidal}")

    result_composite_simpson = composite_simpson_rule(f, a, b, n)
    print(f"result (composite simpson) = {result_composite_simpson}")
