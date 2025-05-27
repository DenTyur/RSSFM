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

N = 128
prs.dx = prs.dy = 0.5
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
prs.b_V = 7.5


# ========================================================================================
f = open("TISE2D_out.txt", "a")
f.write("START: " + str(datetime.datetime.now()) + "\n")
print("devise:", device)
f.write("device: " + str(device) + "\n")
# =====================================================================================


x = prs.dy * (np.arange(N) - 0.5 * N)
y = x
X, Y = np.meshgrid(x, y, indexing="ij")


def get_potential(x, y):
    V = (
        -1.0 / np.sqrt(x**2 + prs.a_V**2)
        - 1.0 / np.sqrt(y**2 + prs.a_V**2)
        + 1.0 / np.sqrt((y - x) ** 2 + prs.b_V**2)
    )
    return V


V = get_potential(X, Y)


print("V.shape = ", V.shape)
f.write("V.shape = " + str(V.shape) + "\n")

diag = np.ones([N])
diags = np.array([diag, -2 * diag, diag])
D = sp.sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
T = -1 / (2 * prs.dx**2) * sp.sparse.kronsum(D, D)
U = sp.sparse.diags(V.reshape(N**2), (0))
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
    return eigenvectors.T[n].reshape((N, N)).cpu()


au_to_ev = 27.2113961317875

# сохранение в.ф. и параметров в файл
psi_bound = get_psi_bound(0)
np.save("psi_initial.npy", psi_bound)
psi_bound = np.load("psi_initial.npy").astype(np.complex64)
np.save("psi_initial.npy", psi_bound)
np.save("x0.npy", x.astype(np.float32))
np.save("x1.npy", x.astype(np.float32))

print(f"E{prs.state_number} = ", eigenvalues * au_to_ev, "eV")
f.write("a_V = " + str(prs.a_V) + "   b_V = " + str(prs.b_V) + "\n")
f.write("E" + str(prs.state_number) + " = " +
        str(eigenvalues * au_to_ev) + " eV\n")

f.write("FINISH: " + str(datetime.datetime.now()) + "\n")
f.write("______________________________________________\n")
f.close()
