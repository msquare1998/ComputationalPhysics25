# ---------------------------------------------------------------------------------------------
#   l1-q5, Gauss-Seidel method (II)
#   Author: Yi-Ming Ding
#   Updated: Feb 19, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve
from l1_q4 import gauss_seidel

A = np.array([
    [7, 2, 1, -2],
    [9, 15, 3, -2],
    [-2, -2, 11, 5],
    [1, 3, 2, 13],
])

b = np.array([4, 7, -1, 0])

if __name__ == "__main__":
    x_itr = gauss_seidel(A, b, tol=1e-5, max_iter=5000)
    print(f"x_itr = {x_itr}")
    print(f"Reference solutions: {solve(A, b)}")