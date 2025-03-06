# ---------------------------------------------------------------------------------------------
#   l2-q4, Newton method
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
#   Reference:
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
#       https://docs.jax.dev/en/latest/_autosummary/jax.grad.html#jax.grad
# ---------------------------------------------------------------------------------------------
from scipy import optimize
from jax import grad

f = lambda x: x**3 + 2 * x**2 + 3 * x - 1
f_prime = lambda x: 3 * x**2 + 4 * x + 3    # manually write the derivative
f_grad = grad(f)    # (recommended) automatic differentiation with jax

if __name__ == "__main__":
    x0 = 0.5
    root = optimize.newton(f, x0=x0, fprime=f_prime)
    print(f"Approximate root (manual f'): {root:.7f}")

    root = optimize.newton(f, x0=x0, fprime=f_grad)
    print(f"Approximate root (autodiff f'): {root:.7f}")