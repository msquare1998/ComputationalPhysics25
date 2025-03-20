# ---------------------------------------------------------------------------------------------
#   l4-q2, Numerical differentiation (Simpson method)
#   Author: Yi-Ming Ding
#   Updated: Mar 12, 2025
# ---------------------------------------------------------------------------------------------
from jax.numpy import sin, pi
from jax import grad
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve

f = lambda x: sin(x)
f_prime_ad = lambda x: grad(f)(x)
a, b = 0, 2 * pi

def tridiagonal_matrix(n):
    A = np.zeros((n, n))
    np.fill_diagonal(A, 4)  # diagonal
    np.fill_diagonal(A[:-1, 1:], 1)  # sub-diagonal
    np.fill_diagonal(A[1:, :-1], 1)  # sub-diagonal
    return A

if __name__ == "__main__":
    n = 65
    x_data = np.linspace(a, b, n + 1)       # x_0, x_1, ..., x_n
    dx = x_data[1] - x_data[0]
    m_0, m_n = f_prime_ad(x_data[0]), f_prime_ad(x_data[-1])
    f_data = np.array([f(x_data[k + 2]) - f(x_data[k]) for k in range(n - 1)])

    bias_term = np.zeros(n - 1)
    bias_term[0], bias_term[-1] = m_0, m_n

    A = tridiagonal_matrix(n - 1)
    b = 3 / dx * f_data - bias_term
    m = solve(A, b) # solve the linear equations

    plt.plot(x_data[1:-1], m, label="Simpson method", linestyle="", marker="o")
    plt.plot(x_data, [f_prime_ad(x) for x in x_data], linestyle="-", label="Automatic differentiation")
    plt.legend()
    plt.show()