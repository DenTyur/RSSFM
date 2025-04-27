import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import time
import os
import gc
from matplotlib.colors import LogNorm

basedir = os.path.abspath(os.getcwd())
src_dir = os.path.abspath(os.path.join(basedir, ".."))

x = np.load(src_dir + "/arrays_saved/p0.npy")
y = np.load(src_dir + "/arrays_saved/p1.npy")
t = np.load(src_dir + "/arrays_saved/time_evol/t.npy")

X, Y = np.meshgrid(x, y, indexing="ij")

if not os.path.exists(src_dir + "/imgs/time_evol/psi_x"):
    os.makedirs(src_dir + "/imgs/time_evol/psi_x")

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), layout="constrained")

for i in range(len(t)):
    ts = time.time()
    psi = np.load(src_dir + f"/arrays_saved/time_evol/psi_p/psi_p_t_{i}.npy")
    if i == 0:
        psi0 = psi.copy()
    axs.set(
        aspect="equal",
        title=f"external |psi(px, py)|^2 step={i} of {len(t)}; t = {t[i]:.{5}f} a.u.",
    )
    b = axs.pcolormesh(
        X,
        Y,
        np.abs(psi) ** 2,
        cmap=cm.jet,
        shading="auto",
        norm=LogNorm(vmin=1e-5, vmax=1),
    )
    cb = plt.colorbar(b, ax=axs)
    fig.savefig(src_dir + f"/imgs/time_evol/psi_p/psi_p_t_{i}.png")
    axs.clear()
    cb.remove()
    gc.collect()
    print(f"step {i} of {len(t)}; time of step = {(time.time()-ts):.{5}f}")
