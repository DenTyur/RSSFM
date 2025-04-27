import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import time
import os
import gc

basedir = os.path.abspath(os.getcwd())
src_dir = os.path.abspath(os.path.join(basedir, ".."))

x = np.load(src_dir + "/arrays_saved/x0.npy")
y = np.load(src_dir + "/arrays_saved/x1.npy")
t = np.load(src_dir + "/arrays_saved/time_evol/volkov/t.npy")

X, Y = np.meshgrid(x, y, indexing="ij")

if not os.path.exists(src_dir + "/imgs/time_evol/volkov/psi_x"):
    os.makedirs(src_dir + "/imgs/time_evol/volkov/psi_x")

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), layout="constrained")

for i in range(len(t)):
    ts = time.time()
    psi = np.load(src_dir + f"/arrays_saved/time_evol/volkov/psi_x_t_{i}.npy")
    axs.set(
        aspect="equal",
        title=f"|volkov(x, y)|^2 step={i} of {len(t)}; t = {t[i]:.{5}f} a.u.",
    )
    b = axs.pcolormesh(
        X,
        Y,
        # np.abs(psi) ** 2,
        psi.real,
        cmap=cm.jet,
        shading="auto",
        norm=LogNorm(vmin=1e-2, vmax=1),
    )
    print(np.min(np.abs(psi) ** 2), np.max(np.abs(psi) ** 2))
    cb = plt.colorbar(b, ax=axs)
    fig.savefig(src_dir + f"/imgs/time_evol/volkov/psi_x_t_{i}.png")
    axs.clear()
    cb.remove()
    gc.collect()
    print(f"step {i} of {len(t)}; time of step = {(time.time()-ts):.{5}f}")
