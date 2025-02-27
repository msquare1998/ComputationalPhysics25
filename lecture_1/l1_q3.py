# ---------------------------------------------------------------------------------------------
#   l1-q3, Banded matrix
#   Author: Yi-Ming Ding
#   Updated: Feb 19, 2025
#   Reference:
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
# ---------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve_banded

a = np.array([
    [1, 1, 0, 0, 0,],
    [1, 2, 1, 0, 0,],
    [0, 1, 3, 1, 0,],
    [0, 0, 1, 4, 1,],
    [0, 0, 0, 1, 5,],
])

b = np.array([3, 8, 15, 24, 29])

# row-major banded storage in LAPACK
ab = np.array([
    [0, 1, 1, 1, 1,],
    [1, 2, 3, 4, 5,],
    [1, 1, 1, 1, 0,]
])

l, u = 1, 1    # number of non-zero lower and upper diagonals

if __name__ == "__main__":
    x = solve_banded((l, u), ab, b)
    print(f"x = {x}")