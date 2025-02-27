# ---------------------------------------------------------------------------------------------
#   l1-q2, LU decomposition
#   Author: Yi-Ming Ding
#   Updated: Feb 19, 2025
#   Reference:
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html
# ---------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import lu_factor, lu_solve

a = np.array([
    [2, 2, 3,],
    [4, 7, 7,],
    [-2, 4, 5,]
])

b = np.array([3, 1, -7,])

if __name__ == "__main__":
    lu, piv = lu_factor(a)  # partial pivoting for improving numerical stability
    print(f"LU = \n{lu}")   # upper triangle for U; lower triangle for L (with 1 at the diagonal line)
    print(f"Pivot indices:")
    for i, p in enumerate(piv):
        print(f"{i} <-> {piv[i]}", end=", ")
    x = lu_solve((lu, piv), b)
    print(f"\nx = {x}")