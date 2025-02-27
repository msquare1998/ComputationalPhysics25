# ---------------------------------------------------------------------------------------------
#   l1-q4, Gauss-Seidel method
#   Author: Yi-Ming Ding
#   Updated: Feb 19, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve

def gauss_seidel(A: np.ndarray, b: np.ndarray, tol: float, max_iter: int, x0: np.ndarray=None) -> np.ndarray:
    """
    Gauss-Seidel method for solving Ax = b

    :param A:   the coefficient matrix A (must be square).
    :param b:   the right-hand side vector b.
    :param tol: the required precision (tolerance for convergence).
    :param max_iter: maximum iteration count
    :param x0: initial guess for x
    :return: the solution vector x
    """
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()    # for mutable object, ```.copy()``` or shallow copy does not change x0 if we modify x (not a reference)

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # use new values
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # use old values
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new

    raise ValueError(f"The algorithm did not converge within {max_iter} iterations. Final x: {x}")

A = np.array([
    [5, -2, 1],
    [1, 5, -3],
    [2, 1, -5],
])

b = np.array([4, 2, -11])

if __name__ == "__main__":
    x_itr = gauss_seidel(A, b, tol=1e-5, max_iter=5000)
    print(f"x_itr = {x_itr}")
    print(f"Reference solutions: {solve(A, b)}")