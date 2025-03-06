# ---------------------------------------------------------------------------------------------
#   l2-q5, Newton method for system of nonlinear equations
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
#   Reference:
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
#       https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html
# ---------------------------------------------------------------------------------------------
from typing import Callable
import jax.numpy as jnp
from jax import jacfwd
from scipy.optimize import root
import numpy as np

def system(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([
        x[0] - 5 * x[1]**2 + 7 * x[2]**2 + 12,
        3 * x[0] * x[1] + x[0] * x[2] - 11 * x[0],
        2 * x[1] * x[2] + 40 * x[0],
    ])

jacobian = lambda x: jacfwd(system)(x)

def newton_iter(system: Callable[[jnp.ndarray], jnp.ndarray], x0: jnp.ndarray, tol: float=1e-5, max_itr: int=500) -> tuple[jnp.ndarray]:
    """
    :param system: the system of nonlinear equations
    :param x0: the initial guess
    :param tol: the convergence tolerance
    :param max_itr: the maximum number of iterations
    :return: the solution
    """
    x = x0.copy()
    for i in range(max_itr):
        A = jacobian(x)
        B = system(x)
        delta = jnp.linalg.solve(A, -B)

        x = x + delta
        if jnp.linalg.norm(delta) < tol:
            return x

    raise ValueError("Max iterations exceeded, no convergence.")

if __name__ == "__main__":
    x0 = jnp.array([1.5, 5.5, -2.])        # you may need many trials to converge with newton method

    sol_pw = root(system, x0=np.array(x0), method='hybr')  # e.g. Powell's method, for comparison
    print(f"Approximate solutions with Powell's method: {sol_pw.x}")

    sol = newton_iter(system, x0)
    print(f"Approximate solutions with Newton method: {sol}")
    print(f"Check equations: f = {system(sol)}")