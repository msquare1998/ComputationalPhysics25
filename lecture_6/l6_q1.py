# -----------------------------------------------------------------------------------------------------
#   l6-q1, up-wind scheme for the convection equation
#   Author: Yi-Ming Ding
#   Updated: Mar 26, 2025
#   Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Parameter
# -------------------------
L = 15        # Spatial length
dx = 0.1      # Spatial step size
dt = 0.05     # Time step size
a = -1.0      # Wave speed
x = np.arange(-L + dx, 0 + dx, dx)  # Spatial grid
n = len(x)    # Number of spatial grid points

# ---------------------------------
# Initial condition (square wave)
# ---------------------------------
r = a * dt / dx
u = np.zeros(n)
u[n - 20:n - 10] = 1
u0 = u.copy()

def up_wind_scheme_update(frame):
    global u
    u[:-1] = (1 + r) * u[:-1] - r * u[1:]   # update
    line.set_ydata(u)
    return line,

if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_title("Advection equation simulation using the upwind scheme")

    # Initialize the curve
    line, = ax.plot(x, u, 'b', linewidth=2, label="Propagating wave")
    ax.plot(x, u0, 'r', linewidth=2, label="Initial square wave")

    ax.set_xlim(-15, 0)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()

    # Run the animation
    ani = animation.FuncAnimation(fig, up_wind_scheme_update, frames=200, interval=50, blit=True)
    plt.show()