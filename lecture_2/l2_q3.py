# ---------------------------------------------------------------------------------------------
#   l2-q3, fixed-point iteration
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
# ---------------------------------------------------------------------------------------------
from typing import Callable
from numpy import sqrt

"""
    For f(x) = x^2 - x - 1
    we have two choices
        (i) x = sqrt(x+1)
        (ii) x = x^2 - 1
    Notice that f(0) = -1 and f(2) = 5, thus the root is in (0, 5) 
"""
f = lambda x: sqrt(x + 1)
g = lambda x: x**2 - 1

def fixed_point_iter(f: Callable[[float], float], x0: float, tol: float=1e-5, max_itr: int=500) -> float:
    """
    :param f: the equation to solve
    :param x0: the initial guess
    :param tol: the tolerance
    :param max_itr: the maximum number of iterations
    :return: the solution
    """
    for _ in range(max_itr):
        x1 = f(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1         # update x0
    raise ValueError("Max iterations exceeded, no convergence.")

if __name__ == "__main__":
    x0 = 0.5
    root = fixed_point_iter(g, x0)
    print(f"Approximate root: {root:.7f}")