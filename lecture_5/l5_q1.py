# ---------------------------------------------------------------------------------------------
#   l1-q1, Euler method and predictor-corrector method for solving y' = y - 2x / y, with y(0) = 1
#   Author: Yi-Ming Ding
#   Updated: Mar 19, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

y_exact = lambda x: (1 + 2 * x) ** (0.5)
f = lambda x, y: y - 2 * x / y      # the differential equation y' = y - 2x / y

y0 = 1  # initial condition
x_range = (0, 1)    # range of x
h = 0.05  # Step size

def euler_method(f, y0: float, x_range: tuple, h: float) -> tuple:
    """
    :param f: the differential equation
    :param y0: the initial value of y(0)
    :param x_range: the range of x values
    :param h: the step size
    :return:
    """
    x_values = np.arange(x_range[0], x_range[1] + h, h)
    y_values = np.zeros_like(x_values)
    y_values[0] = y0

    for i in range(1, len(x_values)):
        y_values[i] = y_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1])

    return x_values, y_values

def predictor_corrector_method(f, y0, x_range, h):
    x_values = np.arange(x_range[0], x_range[1] + h, h)
    y_values = np.zeros_like(x_values)
    y_values[0] = y0

    for P in range(1, len(x_values)):
        x_P, y_P = x_values[P - 1], y_values[P - 1]
        y_pred = y_P + h * f(x_P, y_P)  # 预测
        y_values[P] = y_P + (h / 2) * (f(x_P, y_P) + f(x_values[P], y_pred))  # 校正

    return x_values, y_values

if __name__ == "__main__":
    x_vals, y_vals = euler_method(f, y0, x_range, h)    # Solve using Euler's method
    x_vals_, y_vals_ = predictor_corrector_method(f, y0, x_range, h)  # Solve using Euler's method

    plt.plot(x_vals, y_vals, label="Euler Method", marker="o", linestyle="")
    plt.plot(x_vals_, y_vals_, label="Predictor-Corrector Method", marker="s", linestyle="")
    plt.plot(x_vals, [y_exact(x) for x in x_vals], label="Exact", zorder=-10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
