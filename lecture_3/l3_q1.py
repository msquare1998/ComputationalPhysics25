# ---------------------------------------------------------------------------------------------
#   l3-q1, Three-point parabolic interpolation
#   Author: Yi-Ming Ding
#   Updated: Mar 5, 2025
# ---------------------------------------------------------------------------------------------
import numpy as np

x_node = np.array([0, 1, 2])
y_node = np.array([8, -7.5, -18])

def func_inv(y: float) -> float:
    l0 = x_node[0] * (y - y_node[1]) * (y - y_node[2]) / ((y_node[0] - y_node[1]) * (y_node[0] - y_node[2]))
    l1 = x_node[1] * (y - y_node[0]) * (y - y_node[2]) / ((y_node[1] - y_node[0]) * (y_node[1] - y_node[2]))
    l2 = x_node[2] * (y - y_node[0]) * (y - y_node[1]) / ((y_node[2] - y_node[0]) * (y_node[2] - y_node[1]))
    return float(l0 + l1 + l2)

if __name__ == "__main__":
    x = func_inv(0)
    print(f"x = {x}")