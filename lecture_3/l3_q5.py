# ------------------------------------------------------------------------------------------------------------
#   l3-q5, Linear Fit
#   Author: Yi-Ming Ding
#   Updated: Mar 5, 2025
#   Reference： https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a * x + b

x = np.array([0.5, 1.2, 2.1, 2.9, 3.6, 4.5, 5.7])
y = np.array([2.81, 3.24, 3.80, 4.30, 4.73, 5.29, 6.03])

if __name__ == "__main__":
    params, covariance = curve_fit(linear_func, x, y)   # fitting
    a, b = params
    errors = np.sqrt(np.diag(covariance))
    print(f"Fitted parameters:")
    print(f"a = {a:.2f} ± {errors[0]:.2f}")
    print(f"b = {b:.2f} ± {errors[1]:.2f}")

    plt.title("Linear Fit to Data")
    plt.scatter(x, y, label="Data points", color='blue')
    plt.plot(x, linear_func(x, *params), label=f"Fitted line: y = {a:.2f}x + {b:.2f}", color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()