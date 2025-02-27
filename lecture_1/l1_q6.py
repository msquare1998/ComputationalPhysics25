# ---------------------------------------------------------------------------------------------
#   l1-q6, Successive over-relaxation (SOR)
#   Author: Yi-Ming Ding
#   Updated: Feb 19, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
from scipy.linalg import solve
from l1_q4 import gauss_seidel

def sor(A: np.ndarray, b: np.ndarray, omega: float, tol: float, max_iter: int, x0: np.ndarray=None) -> np.ndarray:
    """
    Successive Over-Relaxation (SOR) method for solving Ax = b.

    :param A:       the coefficient matrix A (must be square)
    :param b:       the right-hand side vector b
    :param omega:   the relaxation factor (1 < omega < 2 for acceleration would be better)
    :param tol:     the required precision (tolerance for convergence)
    :param max_iter: maximum iteration count.
    :param x0:      initial guess for x
    :return:        the solution vector x
    """
    assert A.shape[0] == A.shape[1]
    assert omega > 1 and omega < 2
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Use new values
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Use old values
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new

    raise ValueError(f"The algorithm did not converge within {max_iter} iterations. Final x: {x}")

A = np.array([
    [1, 1, 0, 0, 0],
    [1, 2, 1, 0, 0],
    [0, 1, 3, 1, 0],
    [0, 0, 1, 4, 1],
    [0, 0, 0, 1, 5]
])

b = np.array([2, 4, 5, 6, 6])

if __name__ == "__main__":
    x_sor = sor(A, b, omega=1.5, tol=1e-6, max_iter=10000)
    print(f"x_sor = {x_sor}")
    print(f"x_gs = {gauss_seidel(A, b, tol=1e-6, max_iter=10000)}")
    print(f"Reference solutions: {solve(A, b)}")