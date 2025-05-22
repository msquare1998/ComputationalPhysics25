"""*************************************************
    DMRG for 1D S=1/2 Heisenberg model
    Author: Yi-Ming Ding
    Email: dingyiming@westlake.edu.cn
    Updated: May 21, 2025
*************************************************"""
from tenpy.networks.mps import MPS
from tenpy import models
from tenpy.algorithms import dmrg

def DMRG_Heisenberg_chain(L, chi):
    model_params = dict(L=L, S=0.5, Jx=1.0, Jy=1.0, Jz=1.0, bc_MPS='finite', conserve="None")
    M = models.spins.SpinChain(model_params)
    product_state = ["up"] * M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': None,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10
        },
        'combine': True
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info["E"]
    return E, psi, M

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    L = 20
    chi = 10
    E, _, __ = DMRG_Heisenberg_chain(L, chi)
    print(f"Energy: {E}")