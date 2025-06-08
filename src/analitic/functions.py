import numpy as np

a0 = 5.2918e-9
m = 9.1094e-28
eq = 4.8e-10
h = 1.0546e-27
c = 2.9979e10
tat = 2.418e-17
Eat = 1.714e7
Iat = 4.3597e-11

# Functions transform units


def J_vtcm2_to_E_au(J_vtcm2):
    return np.sqrt((4 * np.pi * J_vtcm2 * 10**7) / c) / Eat


def eV_to_au(Iev):
    Iat = 4.3597e-11
    return (Iev * 1.60217733e-12) / Iat
