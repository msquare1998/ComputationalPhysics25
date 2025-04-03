# -------------------------------------------------------------------
#   Author: Yi-Ming Ding
#   Updated: Apr 2, 2025
# -------------------------------------------------------------------
import numpy as np

Pauli_i = np.eye(2).astype(np.complex128)

Pauli_x = np.array([
    [0, 1],
    [1, 0]
]).astype(np.complex128)

Pauli_y = np.array([
    [0, -1j],
    [1j, 0]]
).astype(np.complex128)

Pauli_z = np.array([
    [1, 0],
    [0, -1]]
).astype(np.complex128)

S_x = 0.5 * Pauli_x
S_y = 0.5 * Pauli_y
S_z = 0.5 * Pauli_z