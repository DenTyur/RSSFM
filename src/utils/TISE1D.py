import datetime
import json
import time
import types
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch as th
from matplotlib import cm

warnings.filterwarnings("ignore")
# device = th.device("cuda" if th.cuda.is_available() else "cpu")
device = th.device("cpu")

# == ПАРАМЕТРЫ ==

prs = types.SimpleNamespace()

prs.Nn = None
prs.N = 128
prs.dx = 0.5
prs.state_number = 0
# br:
# a_V = 1.66
# b_V = 2.6
# c_V = 4.8
# prs.c2_V = 1.0
# c1 = 0.48
# c2 = 2
# const = 1

prs.a_V = 3.15
prs.name_of_psi_array = "model_6ev_coulomb.hdf5"


# ========================================================================================
f = open("TISE1D_out.txt", "a")
f.write("START: " + str(datetime.datetime.now()) + "\n")
print("devise:", device)
f.write("device: " + str(device) + "\n")
# =====================================================================================


x = prs.dx * (np.arange(prs.N) - 0.5 * prs.N)


def get_potential(x):
    V = -1.0 / np.sqrt(x**2 + prs.a_V**2)
    return V


V = get_potential(x)

print("V.shape = ", V.shape)
f.write("V.shape = " + str(V.shape) + "\n")

diag = np.ones([prs.N])
diags = np.array([diag, -2 * diag, diag])
D = sp.sparse.spdiags(diags, np.array([-1, 0, 1]), prs.N, prs.N)
T = -1 / (2 * prs.dx**2) * D  # sp.sparse.kronsum(D, D)
U = sp.sparse.diags(V.reshape(prs.N), (0))
H = T + U

start = time.time()
H = H.tocoo()
H = th.sparse_coo_tensor(
    indices=th.tensor([H.row, H.col]), values=th.tensor(H.data), size=H.shape
).to(device)

print("convertation time = ", time.time() - start)
f.write("convertation time = " + str(time.time() - start) + "\n")

start = time.time()
eigenvalues, eigenvectors = th.lobpcg(H, k=prs.state_number + 1, largest=False)
print("eiginevalues time =", time.time() - start)
f.write("eiginevalues time = " + str(time.time() - start) + "\n")


def get_psi_bound(n):
    return eigenvectors.T[n].reshape((prs.N)).cpu()


au_to_ev = 27.2113961317875
print("E0 = ", eigenvalues, "au")

# сохранение в.ф. и параметров в файл
with h5py.File(f"{prs.name_of_psi_array}", "w") as f5:
    f5.create_dataset("prs", data=json.dumps(prs.__dict__))
    f5.create_dataset("x", data=x)
    f5.create_dataset("E_ev", data=eigenvalues * au_to_ev)
    for i in range(prs.state_number + 1):
        psi_bound = get_psi_bound(i)
        f5.create_dataset(f"psi_{i}", data=psi_bound)
np.save("psi_initial.npy", psi_bound[0])
np.save("x0.npy", x)

print(f"E{prs.state_number} = ", eigenvalues * au_to_ev, "eV")
f.write("a_V = " + str(prs.a_V) + "\n")
f.write("E" + str(prs.state_number) + " = " +
        str(eigenvalues * au_to_ev) + " eV\n")

f.write("Name of psi-array: " + prs.name_of_psi_array + ".npy\n")
f.write("FINISH: " + str(datetime.datetime.now()) + "\n")
f.write("______________________________________________\n")
f.close()
