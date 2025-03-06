# ---------------------------------------------------------------------------------------------
#   l2-q5, monte carlo algorithm for system of nonlinear equations
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
# ---------------------------------------------------------------------------------------------
import jax.numpy as jnp
import numpy as np
from jax.numpy import exp

np.random.seed(42)  # set the seed of pseudo random number generator

def system(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([
        x[0] - 5 * x[1]**2 + 7 * x[2]**2 + 12,
        3 * x[0] * x[1] + x[0] * x[2] - 11 * x[0],
        2 * x[1] * x[2] + 40 * x[0],
    ])

def loss_func(x: jnp.ndarray) -> float:
    residuals = system(x)
    return float(jnp.sum(residuals ** 2))

def rand_move(lr: float, vec_length: int) -> jnp.ndarray:
    vec = np.random.uniform(-1, 1, vec_length)
    vec = vec / jnp.linalg.norm(vec) * lr
    return jnp.array(vec)

def monte_carlo(loss_func, x0: jnp.ndarray, beta: float =10, lr: float=1e-2, tol: float = 1e-3, max_itr: int = 5000) -> tuple[jnp.ndarray]:
    """
    :param loss_func: the loss function
    :param x0: the initial guess
    :param beta: the temperature factor, which should be chosen carefully
    :param lr: the learning rate, or the step of each move
    :param tol: the tolerance
    :param max_itr: maximum number of iterations
    :return: the final solution
    """
    x = x0.copy()
    vec_length = x.shape[0]
    for t in range(max_itr):
        move = rand_move(lr, vec_length)    # randomly choose the next move
        loss_diff = loss_func(x + move) - loss_func(x)  # compute the difference

        # standard Metropolis algorithm here
        if np.random.rand() < min(1.0, float(exp(-beta * loss_diff))):
            x = x + move

        loss = loss_func(x)
        if t % 250 == 0:
            print(f"t = {t}, loss = {loss}")

        if loss < tol:
            print(f"Final loss = {loss}")
            return x

    raise Exception("Maximum iterations reached without convergence.")

if __name__ == '__main__':
    x0 = jnp.array([2.0, 5.0, -3.0])
    sol = monte_carlo(loss_func, x0, beta=10, tol=1e-2, lr=1e-2, max_itr=10000)
    print("Approximate solution:", sol)