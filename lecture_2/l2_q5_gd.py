# ---------------------------------------------------------------------------------------------
#   l2-q5, gradient descent for system of nonlinear equations
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
# ---------------------------------------------------------------------------------------------
import jax.numpy as jnp
from jax import grad

def system(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([
        x[0] - 5 * x[1]**2 + 7 * x[2]**2 + 12,
        3 * x[0] * x[1] + x[0] * x[2] - 11 * x[0],
        2 * x[1] * x[2] + 40 * x[0],
    ])

def loss_func(x: jnp.ndarray) -> float:
    residuals = system(x)
    return jnp.sum(residuals ** 2)

grad_func = grad(loss_func)

def steepest_descent(loss_func, grad_func, x0: jnp.ndarray, tol: float = 1e-6, max_itr: int = 1000) -> tuple[jnp.ndarray]:
    """
    :param loss_func: the loss function
    :param grad_func: the gradient function of the loss function
    :param x0: the initial guess
    :param tol: the tolerance
    :param max_itr: maximum number of iterations
    :return: the final solution
    """
    x = x0.copy()
    for t in range(max_itr):
        gradient = grad_func(x)   # the gradient
        alpha = loss_func(x) / jnp.sum(gradient ** 2)  # compute alpha (the learning rate)
        x = x - alpha * gradient
        loss = loss_func(x)
        if t % 100 == 0:
            print(f"t = {t}, loss = {loss}")
        if loss < tol:
            return x
    raise Exception("Maximum iterations reached without convergence.")

if __name__ == "__main__":
    x0 = jnp.array([2.0, 5.0, -3.0])
    sol = steepest_descent(loss_func, grad_func, x0)
    print("Approximate solution:", sol)