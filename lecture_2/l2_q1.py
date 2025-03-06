# ---------------------------------------------------------------------------------------------
#   l2-q1, bisection method
#   Author: Yi-Ming Ding
#   Updated: Feb 26, 2025
#   Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html
# ---------------------------------------------------------------------------------------------
from scipy import optimize
from numpy import exp, log

f = lambda x: exp(x) * log(x) - x**2  # the function
a, b = 1, 2     # the interval

if __name__ == "__main__":
    root = optimize.bisect(f, a, b)
    print(f"Approximate root: {root:.7f}")