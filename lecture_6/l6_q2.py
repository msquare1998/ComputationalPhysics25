# -----------------------------------------------------------------------------------------------------
#   l6-q2, FTCS scheme for the convection equation
#   Author: Yi-Ming Ding
#   Updated: Mar 26, 2025
# -----------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

N = 100  # Number of spatial grid points
L = 1.0  # Length of the computational domain
h = L / N  # Spatial step size
c = 1.0  # Wave speed
tau = 0.001  # Time step size
coeff = -c * tau / (2.0 * h)  # coefficient
nStep = 1000  # Number of time steps
nplots = 50  # Number of plots
plotStep = nStep // nplots  # Record data every plotStep

# Generate spatial grid
x = (np.arange(1, N + 1) - 0.5) * h - L / 2

# Initial condition: Gaussian-cosine wave packet
sigma = 0.1
k_wave = np.pi / sigma
a = np.cos(k_wave * x) * np.exp(-x ** 2 / (2 * sigma ** 2))
a_init = a.copy()

# Periodic boundary conditions
ip = np.roll(np.arange(N), -1)  # i+1 (right shift)
im = np.roll(np.arange(N), 1)  # i-1 (left shift)

# Store plotting data
tplot = [0]
aplot = [a.copy()]

if __name__ == "__main__":
    # ------------------------------------------
    #   FTCS scheme
    # ------------------------------------------
    for iStep in range(nStep):
        a[:] = a + coeff * (a[ip] - a[im])  # FTCS update

        # Record data every plotStep
        if iStep % plotStep == 0:
            aplot.append(a.copy())
            tplot.append(tau * iStep)

    aplot = np.array(aplot)
    tplot = np.array(tplot)

    # ------------------------------------------
    #   Plotting
    # ------------------------------------------
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, a_init, '-', label="Initial")
    ax1.plot(x, a, '--', label="Final")
    ax1.set_xlabel('x')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.set_title('FTCS Method: Initial vs Final State')
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X, T = np.meshgrid(x, tplot)
    ax2.plot_surface(T, X, aplot, cmap='viridis')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_zlabel('Amplitude')
    ax2.set_title('Wave Propagation using FTCS')

    plt.show()
