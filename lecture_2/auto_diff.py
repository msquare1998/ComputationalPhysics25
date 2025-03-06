# ---------------------------------------------------------------------------------------------
#   Example for automatic differentiation
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
#   Reference: https://docs.jax.dev/en/latest/quickstart.html
# ---------------------------------------------------------------------------------------------
from jax import grad, random
from jax.numpy import sin, cos

# original function
f = lambda x : x**2

# manually defining the grad
f_prime_manual = lambda x: 2 * x

# linear approximation
def f_prime_la(x, dx: float=1e-5):
    return (f(x + dx) - f(x)) / dx

# automatic differentiation
f_prime_ad = lambda x: grad(f)(x)

# a more complicated example
def g(x):
    for i in range(3):
        if i % 3 == 0:
            x = x * x + 3.3 + random.uniform(random.PRNGKey(0), shape=(), minval=0.0, maxval=1.0)
        else:
            x = sin(x + 2) + cos(x + 3)
    return x

def g_prime_la(x, dx: float=1e-6):
    return (g(x + dx) - g(x)) / dx

g_prime_ad = lambda x: grad(g)(x)

if __name__ == "__main__":
    x0 = 2.
    print("f:")
    print(f"\tAuto-diff: {f_prime_ad(x0)}")
    print(f"\tManual: {f_prime_manual(x0)}")
    print(f"\tLinear approx: {f_prime_la(x0)}")

    print(f"g: ")
    print(f"\tAuto-diff: {g_prime_ad(x0):7f}")
    print(f"\tLinear approx: {g_prime_la(x0):7f}")