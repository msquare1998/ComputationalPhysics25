# ---------------------------------------------------------------------------------------------
#   l1-q1, Gaussian Elimination Algorithm
#   Author: Yi-Ming Ding
#   Updated: Feb 19, 2025
#   Reference:
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html
# ---------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve

a = np.array([
    [1, -1, 1, ],
    [5, -4, 3, ],
    [2, 1, 1, ]
])

b = np.array([-4, -12, 11, ])

if __name__ == "__main__":
    x = solve(a, b)  # For general equations, GESV is used
    print(f"x = {x}")