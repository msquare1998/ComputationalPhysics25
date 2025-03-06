# ---------------------------------------------------------------------------------------------
#   l2-q3, fixed-point iteration
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
# ---------------------------------------------------------------------------------------------
from typing import Callable
from numpy import sqrt
import matplotlib.pyplot as plt

"""
    For f(x) = x^2 - x - 1
    we have two choices
        (i) x = sqrt(x+1)
        (ii) x = x^2 - 1
    Notice that f(0) = -1 and f(2) = 5, thus the root is in (0, 5) 
"""
f = lambda x: sqrt(x + 1)
g = lambda x: x**2 - 1

def fixed_point_iter(f: Callable[[float], float], x0: float, tol: float=1e-5, max_itr: int=100) -> (float, list, list):
    x1_list = []

    for i in range(max_itr):
        x1 = f(x0)
        x1_list.append(x1)

        if abs(x1 - x0) < tol:
            return x1, [k for k in range(i + 1)], x1_list

        x0 = x1         # update x0

    #raise ValueError("Max iterations exceeded, no convergence.")
    print("Max iterations exceeded, no convergence.")
    return 777, [k for k in range(max_itr)], x1_list

if __name__ == "__main__":
    x0 = 0.5
    root, step_list, x1_list = fixed_point_iter(g, x0)
    print(f"Approximate root: {root:.7f}")

    plt.plot(step_list, x1_list, marker='o', linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("root")
    plt.show()