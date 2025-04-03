# -------------------------------------------------------------------------------------------
#   Exact diagonalization on S=1/2 Heisenberg model with QuSpin
#   Author: Yi-Ming Ding
#   Updated: Apr 2, 2025
#   References: https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_1d.html
# -------------------------------------------------------------------------------------------
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np

L = 6
J = 1.0

if __name__ == '__main__':
    basis = spin_basis_1d(L,
                          pauli=False,
                          Nup=L // 2,       # total number of spin up or the specification on \sum_j S_j^z; Here it means the zero magnetisation sector
                          pzblock=1,        # symmetries: reflection about the middle of the chain + a simultaneous flip of the sign of the spin-z component
                          )

    lattice = [[J, i, (i + 1) % L] for i in range(L)]

    # static and dynamic lists
    static = [["xx", lattice], ["yy", lattice], ["zz", lattice]]
    dynamic = []
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

    # print and check the block-diagonalized Hamiltonian matrix
    print(f"H =\n{H.todense()}")

    # calculate minimum energy, where "SA" means "Smallest Algebraic"
    E0 = H.eigsh(k=1, which="SA", maxiter=1e4, return_eigenvectors=False)
    print(f"E0 = {E0}")