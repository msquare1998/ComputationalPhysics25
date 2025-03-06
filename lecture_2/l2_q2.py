# ---------------------------------------------------------------------------------------------
#   l2-q2, secant method
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
#   Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
# ---------------------------------------------------------------------------------------------
from scipy import optimize
from numpy import exp, log

f = lambda x: exp(x) * log(x) - x**2
a, b = 1, 2

if __name__ == "__main__":
    root = optimize.newton(f, x0=a, x1=b)   # if we only pass it with x0 and x1, the secant method is used
    print(f"Approximate root: {root:.7f}")