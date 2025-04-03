# -----------------------------------------------------------------------------------------------------
#   l6-q3, Method of lines for diffusion equations
#   Author: Yi-Ming Ding
#   Updated: Mar 26, 2025
#   Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

n = 15  # Number of grid points
dx = np.pi / n
x = np.linspace(dx, np.pi - dx, n - 1)
alpha = 1  # α = a / Δx² = 1
t_span = (0, 40)  # Time interval
t_eval = np.linspace(0, 40, 100)  # Select 100 time points

# Initial condition: u(x,0) = sin(x)
u0 = np.sin(x)

def diffusion_equation(t, u):
    dudt = np.zeros_like(u)
    dudt[0] = -2 * u[0] + u[1]  # Near the left boundary
    dudt[-1] = u[-2] - 2 * u[-1]  # Near the right boundary
    dudt[1:-1] = u[:-2] - 2 * u[1:-1] + u[2:]  # Interior points
    return alpha * dudt

if __name__ == "__main__":
    sol = solve_ivp(diffusion_equation, t_span, u0, t_eval=t_eval)

    # Reconstruct the solution
    uu = np.zeros((len(t_eval), n + 1))  # Includes x=0 and x=pi
    uu[:, 1:n] = sol.y.T   # Assign interior points

    # Generate grid data
    x_full = np.linspace(0, np.pi, n + 1)
    T, X = np.meshgrid(t_eval, x_full)

    # -------------------------------------------------------
    #   Plotting
    # -------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, uu.T, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    ax.set_title('MOL Method for Diffusion Equation')
    plt.show()
