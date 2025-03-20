# ---------------------------------------------------------------------------------------------
#   l4-q3, Numerical integration for I = \int_0^1 [4 / (1 + x^2)] dx
#       Using trapezoidal rule, Simpson 1 / 3 rule, and Simpson 8 / 3 rule
#   Author: Yi-Ming Ding
#   Updated: Mar 12, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
from typing import Callable

f = lambda x: 4 / (1 + x * x)

def trapezoidal_rule(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    :param f: the integrated function
    :param a: left end point
    :param b: right end point
    :param n: number of parameter points
    :return: the value of the integral
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))

def simpson_1_3(f, a, b, n):
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return (h / 3) * (f(x[0]) + 4 * np.sum(f(x[1:-1:2])) + 2 * np.sum(f(x[2:-2:2])) + f(x[-1]))

def simpson_3_8(f, a, b, n):
    if n % 3 != 0:
        n += 3 - (n % 3)
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return (3 * h / 8) * (f(x[0]) + 3 * np.sum(f(x[1:-1:3])) + 3 * np.sum(f(x[2:-1:3])) + 2 * np.sum(f(x[3:-3:3])) + f(x[-1]))

if __name__ == "__main__":
    n = 100 # number of discrete points
    a, b = 0, 1

    result_trapezoidal = trapezoidal_rule(f, a, b, n)
    print(f"result (trapezoidal) = {result_trapezoidal}")

    result_simpson_1_3 = simpson_1_3(f, a, b, n)
    print(f"result (simpson 1 / 3) = {result_simpson_1_3}")

    result_simpson_8_3 = simpson_3_8(f, a, b, n)
    print(f"result (simpson 8 / 3) = {result_simpson_8_3}")