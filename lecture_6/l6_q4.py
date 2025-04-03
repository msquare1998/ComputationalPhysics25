# -----------------------------------------------------------------------------------------------------
#   l6-q4, FTCS for solving the temperature distribution
#   Author: Yi-Ming Ding
#   Updated: Mar 26, 2025
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

L = 1.0         # Rod length
N = 50          # Number of spatial points
h = L / (N - 1)  # Spatial step size
tau = 0.0001   # Time step size
kappa = 0.835  # Thermal diffusion coefficient
coeff = kappa * tau / h**2  # FTCS coefficient
nstep = 300    # number of time steps
nplots = 50    # number of plotting
plot_step = nstep // nplots  # time interval for plotting

# Space grid
x = np.linspace(-L/2, L/2, N)

# ------------------------------------------------------
#   Delta distribution in the middle of the rod
# ------------------------------------------------------
T_delta = np.zeros(N)
T_delta[N // 2] = 1 / h  # an approximate delta function

T_delta_plot = np.zeros((N, nplots))
tplot = np.zeros(nplots)

if __name__ == "__main__":
    # -------------------------
    # FTCS iteration
    # -------------------------
    for step in range(nstep):
        # condition (i)
        T_new = T_delta.copy()
        T_new[1:N-1] = T_delta[1:N-1] + coeff * (T_delta[2:N] + T_delta[:N-2] - 2 * T_delta[1:N-1])
        T_delta = T_new

        if step % plot_step == 0:
            index = step // plot_step
            T_delta_plot[:, index] = T_delta
            tplot[index] = step * tau

    # -------------------------
    #   Plotting
    # -------------------------
    X, T = np.meshgrid(tplot, x)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # First plot (3D surface plot)
    ax1 = axes[0]
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # Using 1x2 grid layout
    ax1.plot_surface(T, X, T_delta_plot, cmap='viridis')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('T(x,t)')

    # Second plot (contour plot)
    ax2 = axes[1]
    cs1 = ax2.contourf(tplot, x, T_delta_plot, levels=20, cmap='viridis')
    fig.colorbar(cs1, ax=ax2)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')

    # Show the plots
    plt.tight_layout()
    plt.show()