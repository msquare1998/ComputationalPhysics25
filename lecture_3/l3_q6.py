# ------------------------------------------------------------------------------------------------------------
#   l3-q6, Polynomial fit to exp(x)
#   Author: Yi-Ming Ding
#   Updated: Mar 5, 2025
#   Reference： https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def poly5(x, a5, a4, a3, a2, a1, a0):
    return a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0

x = np.linspace(-1, 1, 20)
y = np.exp(x)   # generate data first

if __name__ == "__main__":
    params, covariance = curve_fit(poly5, x, y)   # fitting
    a5, a4, a3, a2, a1, a0 = params
    errors = np.sqrt(np.diag(covariance))

    print(f"Fitted parameters:")
    print(f"a5 = {a5:.4f} ± {errors[0]:.4f}")
    print(f"a4 = {a4:.4f} ± {errors[1]:.4f}")
    print(f"a3 = {a3:.4f} ± {errors[2]:.4f}")
    print(f"a2 = {a2:.4f} ± {errors[3]:.4f}")
    print(f"a1 = {a1:.4f} ± {errors[4]:.4f}")
    print(f"a0 = {a0:.4f} ± {errors[5]:.4f}")

    plt.title("5th order polynomial fit")
    plt.plot(x, y, label="exp(x)", linestyle='-', linewidth=4)
    plt.plot(x, poly5(x, *params), label=f"5th order polynomial fit", linestyle=':', linewidth=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()