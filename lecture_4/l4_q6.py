# ---------------------------------------------------------------------------------------------
#   l4-q6, Monte Carlo sampling for estimating pi
#   Author: Yi-Ming Ding
#   Updated: Mar 12, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm

def mc_pi_solver(n_tot: int):
    """
    :param n_tot: number of samples
    :return: the estimation of pi
    """
    x_inside, y_inside = [], []
    x_outside, y_outside = [], []

    n_inside, n_outside = 0, 0

    for _ in tqdm(range(n_tot)):
        x, y = np.random.uniform(0, 1), np.random.uniform(0, 1)
        if sqrt(x * x + y * y) < 1:
            n_inside += 1
            x_inside.append(x)
            y_inside.append(y)
        else:
            n_outside += 1
            x_outside.append(x)
            y_outside.append(y)

    pi_estimated = 4.0 * float(n_inside) / float(n_tot)
    return pi_estimated, x_inside, x_outside, y_inside, y_outside

quarter_circle = lambda x: sqrt(1 - x * x)

if __name__ == "__main__":
    n = int(1e5)    # number of samples
    pi_estimated, x_inside, x_outside, y_inside, y_outside = mc_pi_solver(n)

    plt.title(fr"$n={n}$, $\pi\approx${pi_estimated}")
    plt.plot(x_inside, y_inside, marker="o", color="red", markersize=2.5, linestyle="")
    plt.plot(x_outside, y_outside, marker="s", color="blue", markersize=2.5, linestyle="")

    x_data = np.linspace(0, 1, 100)
    plt.plot(x_data, quarter_circle(x_data), color="black")
    plt.plot(x_data, [0 for x in x_data], color="black")
    plt.plot([0 for x in x_data], x_data, color="black")

    plt.show()