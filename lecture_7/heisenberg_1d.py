# -------------------------------------------------------------------
#   Brute-force exact diagonalization on S=1/2 Heisenberg model
#   Author: Yi-Ming Ding
#   Updated: Apr 2, 2025
# -------------------------------------------------------------------
from numpy import kron, trace
from preset import *
from scipy.linalg import expm

class Heisenberg1D:
    def __init__(self, _l):
        """
        :param _l: the length of the 1D ring
        :param _beta: inverse temperature
        """
        self.l = _l
        self.lattice = np.array([[i, (i + 1) % L] for i in range(self.l)])
        self.num_s = self.l  # number of sites

        # --------------------------------
        #   Make the Hamiltonian
        # --------------------------------
        self.H = np.zeros((2 ** self.num_s, 2 ** self.num_s)).astype(complex)
        for link in self.lattice:
            hxx, hyy, hzz = 1, 1, 1
            for q in range(self.num_s):
                if q in link:
                    hxx, hyy, hzz = kron(S_x, hxx), kron(S_y, hyy), kron(S_z, hzz)
                else:
                    hxx, hyy, hzz = kron(Pauli_i, hxx), kron(Pauli_i, hyy), kron(Pauli_i, hzz)
            self.H += hxx + hyy + hzz

    def get_ground_state_info(self):
        vals, vecs = np.linalg.eigh(self.H)
        energy0 = vals[0]
        assert np.min(vals) == energy0
        vec0 = vecs[:, 0].reshape((-1, 1))
        rho0 = vec0 @ np.transpose(np.conj(vec0))
        return energy0, rho0

    def get_thermal_state_info(self, beta):
        assert beta > 0, "beta must be greater than 0"
        rho = np.array(expm(-beta * self.H))
        rho = rho / trace(rho)  # normalization
        energy = trace(rho @ self.H)
        return energy.real, rho

    def get_correlation_zz_op(self, s0, s1):
        assert s0 != s1
        corr_op = 1
        for q in range(self.num_s):
            if q == s0 or q == s1:
                corr_op = kron(Pauli_z, corr_op)
            else:
                corr_op = kron(Pauli_i, corr_op)
        return corr_op

    def calc_expectation(self, rho, A):
        return (trace(rho @ A)).real

L = 6
beta = 1
s0, s1 = 0, 1

if __name__ == '__main__':
    model = Heisenberg1D(_l=L)

    E0, rho0 = model.get_ground_state_info()
    print(f"Ground state energy = {E0}")

    corr_zz_op = model.get_correlation_zz_op(s0, s1)
    cor_zz = model.calc_expectation(rho0, corr_zz_op)
    print(f"Correlation ZZ (GS) = {cor_zz}")

    E, rho = model.get_thermal_state_info(beta)
    print(f"Energy (beta = {beta}) = {E}")

    cor_zz = model.calc_expectation(rho, corr_zz_op)
    print(f"Correlation ZZ (Thermal) = {cor_zz}")