# -----------------------------------------------------------------------------------------------------
#   l6-q5, Solving Helmholtz equation
#   Author: Yi-Ming Ding
#   Updated: Mar 26, 2025
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

f = lambda x ,y: x ** 2 + y ** 2
g = lambda x ,y: np.sqrt(x)

# Boundary conditions
bx0 = lambda y: y ** 2
bxf = lambda y: 16 * np.cos(y)
by0 = lambda x:  x ** 2
byf = lambda x: 16 * np.cos(x)

def Helmholtz(f, g, bx0, bxf, by0, byf, D, Mx, My, tol, MaxIter):
    """
    :param D: the region for solution: D = (x0, xf, y0, yf)
    :param Mx: number of grid points in x direction
    :param My: number of grid points in y direction
    :param tol: the tolerance
    :param MaxIter: maximum number of iterations
    :return: the solution
    """
    x0, xf, y0, yf = D

    # Discretize the grid
    dx = (xf - x0) / (Mx - 1)
    dy = (yf - y0) / (My - 1)
    x = np.linspace(x0, xf, Mx)
    y = np.linspace(y0, yf, My)

    # Initialize the u matrix and set boundary conditions
    u = np.zeros((Mx, My))

    u[:, 0] = np.array([by0(xx) for xx in x])  # Boundary u(x, 0)
    u[:, -1] = np.array([byf(xx) for xx in x])  # Boundary u(x, yf)
    u[0, :] = np.array([bx0(yy) for yy in y])  # Boundary u(0, y)
    u[-1, :] = np.array([bxf(yy) for yy in y])  # Boundary u(xf, y)

    # Iterative solution for internal points
    for _ in range(MaxIter):
        u_old = u.copy()

        # Update the internal u values
        for i in range(1, Mx - 1):
            for j in range(1, My - 1):
                u[i, j] = (dy ** 2 * (u_old[i + 1, j] + u_old[i - 1, j]) + dx ** 2 * (
                            u_old[i, j + 1] + u_old[i, j - 1]) - dx ** 2 * dy ** 2 * f(x[i], y[j])) / (
                                      2 * (dx ** 2 + dy ** 2) + g(x[i], y[j]) * dx ** 2 * dy ** 2)

        # Check for convergence
        if np.linalg.norm(u - u_old, ord=np.inf) < tol:
            break

    return u, x, y

if __name__ == '__main__':
    D = [0, 4, 0, 4]  #
    Mx = 30  # Number of grid points in the x-direction
    My = 30  # Number of grid points in the y-direction
    MaxIter = 100  # Maximum number of iterations
    tol = 1e-4  # Convergence tolerance

    u, x, y = Helmholtz(f, g, bx0, bxf, by0, byf, D, Mx, My, tol, MaxIter)

    # Plot the results
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u.T, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title('Numerical Solution of the Helmholtz Equation')

    plt.show()