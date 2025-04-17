# -------------------------------------------------------------------------------------------
#   Exact diagonalization on S=1/2 Heisenberg model with QuSpin
#   Author: Yi-Ming Ding
#   Updated: Apr 9, 2025
# -------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

# -----------------------------------------------------------------------
#   J > 0 for antiferromagnetism, and J < 0 for ferromagnetism
# -----------------------------------------------------------------------
L = 20
J = -1.0

lattice = [[J, i, (i + 1) % L] for i in range(L)]   # PBC is required
momenta = np.arange(L)
energies = []

if __name__ == "__main__":
    for moment in momenta:
        basis = spin_basis_1d(L,
                              kblock=int(moment),   # the k-sector
                              Nup=L // 2,   # total number of spin up or the specification on \sum_j S_j^z; Here it means the zero magnetisation sector
                              pzblock=1,    # symmetries: reflection about the middle of the chain + a simultaneous flip of the sign of the spin-z component
                              pauli=False,
                              )
        static = [["xx", lattice], ["yy", lattice], ["zz", lattice]]
        H = hamiltonian(static, [], basis=basis, dtype=np.complex64)
        E = H.eigvalsh()[0]  # take the ground state energy in this momentum sector
        energies.append(E)

    k_vals = 2 * np.pi * momenta / L        # Convert momentum index to actual momentum values
    if J > 0:
        plt.title(fr"AFM 1D Heisenberg Model, $L={L}$")
    else:
        plt.title(fr"FM 1D Heisenberg Model, $L={L}$")
    plt.plot(k_vals, energies, 'o-', label='Dispersion relation')
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")

    plt.xticks(np.linspace(0, 2 * np.pi, L + 1, endpoint=True),
               [f"${k:.1f}\pi$" for k in np.linspace(0, 2, L + 1, endpoint=True)])
    plt.legend()
    plt.show()